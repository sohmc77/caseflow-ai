import threading

import pytest

from agents.base import AgentError
from agents.pipeline import REVIEW_GRAPH, run_case_review
from agents.providers import CompletionRequest, CompletionResponse, MockProvider
from agents.schemas import ActionKind, RequirementStatus

from .conftest import make_extraction


def test_full_review_with_mock(snapshot):
    result = run_case_review(snapshot, MockProvider())

    statuses = {e.requirement_code: e.status for e in result.evaluation.evaluations}
    assert statuses == {
        "identity": RequirementStatus.SATISFIED,
        "job_offer": RequirementStatus.MISSING,
        "proof_of_funds": RequirementStatus.NOT_SATISFIED,
    }
    kinds = {(a.kind, a.document_id, a.requirement_code) for a in result.plan.actions}
    assert (ActionKind.ACCEPT_DOCUMENT, 10, None) in kinds
    assert (ActionKind.REJECT_DOCUMENT, 11, None) in kinds
    assert (ActionKind.REQUEST_DOCUMENT, None, "job_offer") in kinds
    assert not any(a.kind == ActionKind.CHANGE_CASE_STATUS for a in result.plan.actions)
    assert [t.agent for t in result.traces] == [
        "DocumentReviewAgent",
        "DocumentReviewAgent",
        "ChecklistAgent",
        "NextStepAgent",
    ]


class BrokenForDocument(MockProvider):
    """Heuristic mock, except extraction for one document always returns garbage."""

    def __init__(self, broken_document_id: int):
        super().__init__()
        self.broken = broken_document_id

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        task = next(m.content for m in request.messages if m.role == "user")
        if request.output_schema_name == "DocumentExtractionResult" and f'"document_id":{self.broken}' in task:
            return CompletionResponse(content="I think this is a bank statement.", model="broken")
        return super().complete(request)


def test_one_bad_document_is_flagged_not_fatal(snapshot):
    result = run_case_review(snapshot, BrokenForDocument(11))

    assert [f.document_id for f in result.failed_documents] == [11]
    assert result.evaluation.get("proof_of_funds").status == RequirementStatus.NEEDS_REVIEW
    flagged = {a.document_id for a in result.plan.actions if a.kind == ActionKind.FLAG_FOR_REVIEW}
    assert 11 in flagged


def test_prior_accepted_extractions_count_without_re_review(snapshot):
    accepted = snapshot.model_copy(
        update={"documents": [snapshot.documents[0].model_copy(update={"status": "accepted"}), snapshot.documents[1]]}
    )
    prior = {
        10: make_extraction(10, "passport", full_name="Amara Okafor", passport_number="X", expiry_date="2031-01-01")
    }
    result = run_case_review(accepted, MockProvider(), prior)

    assert [e.document_id for e in result.extractions] == [11]  # only the pending doc was re-read
    assert result.evaluation.get("identity").status == RequirementStatus.SATISFIED
    assert not any(a.kind == ActionKind.ACCEPT_DOCUMENT and a.document_id == 10 for a in result.plan.actions)


def test_checklist_failure_aborts_the_review(snapshot):
    class BrokenChecklist(MockProvider):
        def complete(self, request):
            if request.output_schema_name == "ChecklistEvaluation":
                return CompletionResponse(content="{}", model="broken")
            return super().complete(request)

    with pytest.raises(AgentError, match="ChecklistAgent"):
        run_case_review(snapshot, BrokenChecklist())


# ------------------------------------------------------------------ LangGraph behaviour


def test_graph_shape():
    assert set(REVIEW_GRAPH.get_graph().nodes) == {
        "__start__",
        "review_document",
        "evaluate_checklist",
        "plan_next_steps",
        "__end__",
    }


def test_documents_are_reviewed_in_parallel(snapshot):
    """Both Send branches must be in flight at the same time, or the barrier times out."""
    barrier = threading.Barrier(2, timeout=5)

    class Rendezvous(MockProvider):
        def complete(self, request):
            first_turn = not any(m.role == "tool" for m in request.messages)
            if request.output_schema_name == "DocumentExtractionResult" and first_turn:
                barrier.wait()  # raises BrokenBarrierError if the branches run one after another
            return super().complete(request)

    result = run_case_review(snapshot, Rendezvous())
    assert [e.document_id for e in result.extractions] == [10, 11]


def test_no_pending_documents_skips_the_fan_out(snapshot):
    done = snapshot.model_copy(
        update={"documents": [d.model_copy(update={"status": "rejected"}) for d in snapshot.documents]}
    )
    result = run_case_review(done, MockProvider())

    assert [t.agent for t in result.traces] == ["ChecklistAgent", "NextStepAgent"]
    assert {e.status for e in result.evaluation.evaluations} == {RequirementStatus.MISSING}
