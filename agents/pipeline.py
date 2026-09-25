"""Multi-step case review: extract → evaluate → plan.

Each step receives only the *validated* output of the previous one. One bad
document does not sink the whole review: it is recorded as a failure and
flagged for a human, and the rest of the pipeline continues.
"""

from collections.abc import Mapping

from .base import AgentError
from .checklist import ChecklistAgent
from .document_review import DocumentReviewAgent
from .next_step import NextStepAgent
from .providers.base import LLMProvider
from .schemas import (
    CaseReviewResult,
    CaseSnapshot,
    ChecklistInput,
    DocumentExtractionResult,
    DocumentFailure,
    DocumentReviewInput,
    NextStepInput,
)
from .tools import AgentContext


def run_case_review(
    snapshot: CaseSnapshot,
    provider: LLMProvider,
    prior_extractions: Mapping[int, DocumentExtractionResult] | None = None,
) -> CaseReviewResult:
    """Review pending documents, then re-evaluate the whole checklist.

    ``prior_extractions`` are extractions a human already accepted; they count
    toward the checklist without being re-extracted.

    Raises ``AgentError`` if the checklist or planning step cannot produce a
    valid output. A partial plan is worse than none.
    """
    traces = []
    extractions: dict[int, DocumentExtractionResult] = dict(prior_extractions or {})
    fresh: list[DocumentExtractionResult] = []
    failures: list[DocumentFailure] = []

    # Step 1: per-document extraction (only documents awaiting review).
    reviewer = DocumentReviewAgent(provider)
    for doc in snapshot.documents:
        if doc.status != "pending_review":
            continue
        try:
            result = reviewer.run(DocumentReviewInput(document_id=doc.id), AgentContext(snapshot))
        except AgentError as exc:
            traces.append(exc.trace)
            failures.append(DocumentFailure(document_id=doc.id, error=str(exc)))
            continue
        traces.append(result.trace)
        fresh.append(result.output)
        extractions[doc.id] = result.output

    ctx = AgentContext(snapshot=snapshot, extractions=extractions)
    unreviewed = [f.document_id for f in failures]

    # Step 2: checklist evaluation over every validated extraction.
    evaluation = ChecklistAgent(provider).run(
        ChecklistInput(
            case_id=snapshot.id,
            extractions=list(extractions.values()),
            unreviewed_document_ids=unreviewed,
        ),
        ctx,
    )
    traces.append(evaluation.trace)

    # Step 3: proposed actions derived from the validated evaluation.
    plan = NextStepAgent(provider).run(
        NextStepInput(case_id=snapshot.id, evaluation=evaluation.output, unreviewed_document_ids=unreviewed),
        ctx,
    )
    traces.append(plan.trace)

    return CaseReviewResult(
        case_id=snapshot.id,
        extractions=fresh,
        failed_documents=failures,
        evaluation=evaluation.output,
        plan=plan.output,
        traces=traces,
    )
