"""Bridges the Django world and the framework-agnostic agent layer.

    DB rows ──build_snapshot──▶ CaseSnapshot ──run_case_review──▶ CaseReviewResult
                                                                        │
    ProposedAction rows (pending) ◀──────────persist_result─────────────┘

Agents only ever see the snapshot. The single write path is ``persist_result``,
and it writes suggestions, never case state.
"""

import logging

from django.db import IntegrityError, transaction
from django.utils import timezone

from agents.base import AgentError
from agents.pipeline import run_case_review
from agents.providers import LLMProvider, get_provider
from agents.schemas import (
    ActionKind,
    CaseReviewResult,
    CaseSnapshot,
    DocumentExtractionResult,
    DocumentSnapshot,
    ProposedActionDraft,
    RequirementSnapshot,
)

from ..models import AgentRun, Case, ProposedAction
from . import audit

logger = logging.getLogger(__name__)


class ReviewAlreadyRunning(Exception):
    def __init__(self, run: AgentRun):
        super().__init__(f"review {run.pk} is already {run.status}")
        self.run = run


def start_review(case: Case, user, provider_name: str) -> AgentRun:
    """Create a queued run and enqueue it once the transaction commits."""
    from ..tasks import run_case_review_task

    try:
        with transaction.atomic():
            Case.objects.select_for_update().get(pk=case.pk)
            active = case.agent_runs.filter(status__in=[AgentRun.Status.QUEUED, AgentRun.Status.RUNNING]).first()
            if active:
                raise ReviewAlreadyRunning(active)
            run = AgentRun.objects.create(case=case, provider=provider_name, triggered_by=user)
            audit.record(case, "review_requested", user=user, target=run, data={"provider": provider_name})
            transaction.on_commit(lambda: run_case_review_task.delay(run.pk))
    except IntegrityError:  # lost a race against the one_active_run_per_case constraint
        raise ReviewAlreadyRunning(case.agent_runs.filter(status__in=["queued", "running"]).first()) from None
    run.refresh_from_db()
    return run


def build_snapshot(case: Case) -> tuple[CaseSnapshot, dict[int, DocumentExtractionResult]]:
    documents = list(case.documents.all())
    snapshot = CaseSnapshot(
        id=case.pk,
        case_type=case.case_type,
        applicant_name=case.applicant_name,
        status=case.status,
        as_of=timezone.localdate(),
        documents=[
            DocumentSnapshot(id=d.pk, filename=d.original_filename, text=d.text, status=d.status) for d in documents
        ],
        requirements=[
            RequirementSnapshot(
                code=r.code,
                title=r.title,
                accepted_document_types=r.accepted_document_types,
                rules=r.rules,
                status=r.status,
            )
            for r in case.checklist.all()
        ],
    )
    prior = {
        d.pk: DocumentExtractionResult.model_validate(d.extraction)
        for d in documents
        if d.status == d.Status.ACCEPTED and d.extraction
    }
    return snapshot, prior


def execute_review(run_id: int, provider: LLMProvider | None = None) -> AgentRun | None:
    # Atomic claim: a redelivered task finds nothing to do instead of running twice.
    claimed = AgentRun.objects.filter(pk=run_id, status=AgentRun.Status.QUEUED).update(
        status=AgentRun.Status.RUNNING, started_at=timezone.now()
    )
    if not claimed:
        logger.info("agent run %s already claimed; skipping", run_id)
        return None
    run = AgentRun.objects.select_related("case").get(pk=run_id)

    try:
        snapshot, prior = build_snapshot(run.case)
        result = run_case_review(snapshot, provider or get_provider(run.provider), prior)
    except Exception as exc:  # any failure ends the run as FAILED; nothing is proposed
        logger.exception("agent run %s failed", run_id)
        _fail(run, exc)
        return run

    persist_result(run, result)
    return run


@transaction.atomic
def persist_result(run: AgentRun, result: CaseReviewResult) -> None:
    case = Case.objects.select_for_update().get(pk=run.case_id)
    # Only the latest review's suggestions are actionable.
    superseded = (
        ProposedAction.objects.filter(case=case, status=ProposedAction.Status.PENDING)
        .exclude(agent_run=run)
        .update(status=ProposedAction.Status.SUPERSEDED)
    )
    documents = {d.pk: d for d in case.documents.all()}
    requirements = {r.code: r for r in case.checklist.all()}
    extractions = {e.document_id: e for e in result.extractions}

    actions = ProposedAction.objects.bulk_create(
        ProposedAction(
            case=case,
            agent_run=run,
            kind=draft.kind,
            target_document=documents.get(draft.document_id),
            target_requirement=requirements.get(draft.requirement_code),
            rationale=draft.rationale,
            payload=_payload(draft, extractions),
        )
        for draft in result.plan.actions
    )

    run.status = AgentRun.Status.SUCCEEDED
    run.result = result.model_dump(mode="json", exclude={"traces"})
    run.traces = [t.model_dump(mode="json") for t in result.traces]
    run.finished_at = timezone.now()
    run.save()
    audit.record(
        case,
        "review_completed",
        agent="review-pipeline",
        target=run,
        data={
            "proposed_actions": len(actions),
            "superseded_actions": superseded,
            "failed_documents": [f.document_id for f in result.failed_documents],
        },
    )


def _payload(draft: ProposedActionDraft, extractions: dict[int, DocumentExtractionResult]) -> dict:
    """Data the approval will apply. Filled in by code, not copied through the model."""
    if draft.kind == ActionKind.ACCEPT_DOCUMENT:
        return {"extraction": extractions[draft.document_id].model_dump(mode="json")}
    if draft.kind == ActionKind.CHANGE_CASE_STATUS:
        return {"new_status": draft.new_case_status}
    return {"message": draft.message} if draft.message else {}


def _fail(run: AgentRun, exc: Exception) -> None:
    run.status = AgentRun.Status.FAILED
    run.error = f"{type(exc).__name__}: {exc}"
    run.traces = [exc.trace.model_dump(mode="json")] if isinstance(exc, AgentError) else []
    run.finished_at = timezone.now()
    run.save()
    audit.record(run.case, "review_failed", agent="review-pipeline", target=run, data={"error": run.error})
