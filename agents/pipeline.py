"""Multi-step case review as a LangGraph graph: extract → evaluate → plan.

    START ──Send×N──▶ review_document ──┐      (one branch per pending document, run in parallel)
      │                                 ▼
      └── no pending docs ──▶ evaluate_checklist ─▶ plan_next_steps ─▶ END

- Fan-out uses LangGraph's ``Send`` API. Branch results are merged by the
  ``operator.add`` reducers on ``extractions`` / ``failures`` / ``traces``, and
  ``evaluate_checklist`` runs once, after every branch has finished.
- Each step receives only the *validated* output of the previous one.
- One bad document does not sink the review. Its branch records a failure, the
  document is flagged for a human, and the rest continues. If the checklist or
  planning step fails, the error propagates: a partial plan is worse than none.
- The provider is injected through ``config["configurable"]``, not the state,
  so the state only holds domain data.

Run ``python -m agents.visualize`` to print this graph as Mermaid.
"""

import operator
from collections.abc import Mapping
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .base import AgentError
from .checklist import ChecklistAgent
from .document_review import DocumentReviewAgent
from .next_step import NextStepAgent
from .providers.base import LLMProvider
from .schemas import (
    AgentTrace,
    CaseReviewResult,
    CaseSnapshot,
    ChecklistEvaluation,
    ChecklistInput,
    DocumentExtractionResult,
    DocumentFailure,
    DocumentReviewInput,
    NextStepInput,
    NextStepPlan,
)
from .tools import AgentContext

MAX_PARALLEL_DOCUMENTS = 4


class ReviewState(TypedDict, total=False):
    snapshot: CaseSnapshot
    prior_extractions: dict[int, DocumentExtractionResult]
    extractions: Annotated[list[DocumentExtractionResult], operator.add]
    failures: Annotated[list[DocumentFailure], operator.add]
    traces: Annotated[list[AgentTrace], operator.add]
    evaluation: ChecklistEvaluation
    plan: NextStepPlan


class DocumentTask(TypedDict):
    """Payload of one ``Send`` branch."""

    snapshot: CaseSnapshot
    document_id: int


def _provider(config: RunnableConfig) -> LLMProvider:
    return config["configurable"]["provider"]


# ---------------------------------------------------------------- routing


def dispatch_documents(state: ReviewState) -> list[Send] | str:
    """Fan out one review branch per document awaiting review."""
    snapshot = state["snapshot"]
    sends = [
        Send("review_document", DocumentTask(snapshot=snapshot, document_id=doc.id))
        for doc in snapshot.documents
        if doc.status == "pending_review"
    ]
    return sends or "evaluate_checklist"


# ---------------------------------------------------------------- nodes


def review_document(task: DocumentTask, config: RunnableConfig) -> ReviewState:
    agent = DocumentReviewAgent(_provider(config))
    try:
        result = agent.run(DocumentReviewInput(document_id=task["document_id"]), AgentContext(task["snapshot"]))
    except AgentError as exc:
        return {"failures": [DocumentFailure(document_id=task["document_id"], error=str(exc))], "traces": [exc.trace]}
    return {"extractions": [result.output], "traces": [result.trace]}


def evaluate_checklist(state: ReviewState, config: RunnableConfig) -> ReviewState:
    snapshot = state["snapshot"]
    all_extractions = {**state.get("prior_extractions", {}), **{e.document_id: e for e in state.get("extractions", [])}}
    unreviewed = sorted(f.document_id for f in state.get("failures", []))
    result = ChecklistAgent(_provider(config)).run(
        ChecklistInput(
            case_id=snapshot.id,
            extractions=[all_extractions[k] for k in sorted(all_extractions)],
            unreviewed_document_ids=unreviewed,
        ),
        AgentContext(snapshot, all_extractions),
    )
    return {"evaluation": result.output, "traces": [result.trace]}


def plan_next_steps(state: ReviewState, config: RunnableConfig) -> ReviewState:
    snapshot = state["snapshot"]
    all_extractions = {**state.get("prior_extractions", {}), **{e.document_id: e for e in state.get("extractions", [])}}
    result = NextStepAgent(_provider(config)).run(
        NextStepInput(
            case_id=snapshot.id,
            evaluation=state["evaluation"],
            unreviewed_document_ids=sorted(f.document_id for f in state.get("failures", [])),
        ),
        AgentContext(snapshot, all_extractions),
    )
    return {"plan": result.output, "traces": [result.trace]}


# ---------------------------------------------------------------- graph


def build_review_graph():
    graph = StateGraph(ReviewState)
    graph.add_node("review_document", review_document)
    graph.add_node("evaluate_checklist", evaluate_checklist)
    graph.add_node("plan_next_steps", plan_next_steps)

    graph.add_conditional_edges(START, dispatch_documents, ["review_document", "evaluate_checklist"])
    graph.add_edge("review_document", "evaluate_checklist")
    graph.add_edge("evaluate_checklist", "plan_next_steps")
    graph.add_edge("plan_next_steps", END)
    return graph.compile(name="case_review")


REVIEW_GRAPH = build_review_graph()


def run_case_review(
    snapshot: CaseSnapshot,
    provider: LLMProvider,
    prior_extractions: Mapping[int, DocumentExtractionResult] | None = None,
) -> CaseReviewResult:
    """Review pending documents, then re-evaluate the whole checklist.

    ``prior_extractions`` are extractions a human already accepted. They count
    toward the checklist without being re-extracted.

    Raises ``AgentError`` if the checklist or planning step cannot produce a
    valid output.
    """
    final = REVIEW_GRAPH.invoke(
        {"snapshot": snapshot, "prior_extractions": dict(prior_extractions or {})},
        config={"configurable": {"provider": provider}, "max_concurrency": MAX_PARALLEL_DOCUMENTS},
    )
    return CaseReviewResult(
        case_id=snapshot.id,
        extractions=sorted(final.get("extractions", []), key=lambda e: e.document_id),
        failed_documents=sorted(final.get("failures", []), key=lambda f: f.document_id),
        evaluation=final["evaluation"],
        plan=final["plan"],
        traces=final["traces"],
    )
