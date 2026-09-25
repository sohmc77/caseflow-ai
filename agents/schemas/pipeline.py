"""Inputs for each agent step and the combined result of a case review.

Each step's input is built from the *validated* output of the previous step, so
nothing unvalidated ever crosses a step boundary.
"""

from typing import Any, Literal

from .actions import NextStepPlan
from .checklist import ChecklistEvaluation
from .common import Contract
from .extraction import DocumentExtractionResult


class DocumentReviewInput(Contract):
    document_id: int


class ChecklistInput(Contract):
    case_id: int
    extractions: list[DocumentExtractionResult]
    unreviewed_document_ids: list[int] = []


class NextStepInput(Contract):
    case_id: int
    evaluation: ChecklistEvaluation
    unreviewed_document_ids: list[int] = []


class AgentStep(Contract):
    """One entry in an agent's trace, persisted for auditability."""

    type: Literal["tool_call", "tool_error", "validation_error", "output"]
    detail: dict[str, Any] = {}


class AgentTrace(Contract):
    agent: str
    provider: str
    turns: int
    validation_retries: int
    input_tokens: int
    output_tokens: int
    steps: list[AgentStep]


class DocumentFailure(Contract):
    document_id: int
    error: str


class CaseReviewResult(Contract):
    case_id: int
    extractions: list[DocumentExtractionResult]
    failed_documents: list[DocumentFailure]
    evaluation: ChecklistEvaluation
    plan: NextStepPlan
    traces: list[AgentTrace]
