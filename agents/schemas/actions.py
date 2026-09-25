"""Output contract of the NextStepAgent.

The agent never changes a case. It returns a ``NextStepPlan`` of drafts; the
service layer stores each draft as a pending ``ProposedAction`` that a human
must approve before anything happens.
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, ValidationInfo, model_validator

from .checklist import RequirementStatus
from .common import Contract


class ActionKind(StrEnum):
    ACCEPT_DOCUMENT = "accept_document"
    REJECT_DOCUMENT = "reject_document"
    MARK_REQUIREMENT_SATISFIED = "mark_requirement_satisfied"
    REQUEST_DOCUMENT = "request_document"
    FLAG_FOR_REVIEW = "flag_for_review"
    CHANGE_CASE_STATUS = "change_case_status"


# kind -> (required fields, forbidden fields)
_TARGETS: dict[ActionKind, tuple[set[str], set[str]]] = {
    ActionKind.ACCEPT_DOCUMENT: ({"document_id"}, {"requirement_code", "new_case_status", "message"}),
    ActionKind.REJECT_DOCUMENT: ({"document_id"}, {"requirement_code", "new_case_status"}),
    ActionKind.MARK_REQUIREMENT_SATISFIED: (
        {"requirement_code", "document_id"},
        {"new_case_status", "message"},
    ),
    ActionKind.REQUEST_DOCUMENT: ({"requirement_code", "message"}, {"document_id", "new_case_status"}),
    ActionKind.FLAG_FOR_REVIEW: (set(), {"new_case_status"}),
    ActionKind.CHANGE_CASE_STATUS: ({"new_case_status"}, {"document_id", "requirement_code", "message"}),
}


class ProposedActionDraft(Contract):
    kind: ActionKind
    document_id: int | None = None
    requirement_code: str | None = None
    new_case_status: Literal["ready_for_submission"] | None = None
    message: str | None = Field(default=None, description="Draft text for the applicant (request_document only).")
    rationale: str = Field(min_length=10, description="Why the agent proposes this, for the reviewer.")

    @model_validator(mode="after")
    def _targets_match_kind(self) -> Self:
        required, forbidden = _TARGETS[self.kind]
        present = {name for name in required | forbidden if getattr(self, name) is not None}
        if missing := sorted(required - present):
            raise ValueError(f"{self.kind} requires {missing}")
        if extra := sorted(forbidden & present):
            raise ValueError(f"{self.kind} must not set {extra}")
        if self.kind == ActionKind.FLAG_FOR_REVIEW and not (self.document_id or self.requirement_code):
            raise ValueError("flag_for_review needs a document_id or requirement_code")
        return self

    @property
    def key(self) -> tuple[str, int | None, str | None]:
        return (self.kind, self.document_id, self.requirement_code)


class NextStepPlan(Contract):
    summary: str = Field(min_length=1)
    actions: list[ProposedActionDraft] = Field(default=[], max_length=20)

    @model_validator(mode="after")
    def _grounded_in_upstream(self, info: ValidationInfo) -> Self:
        """Every draft must be justified by the validated output of earlier steps."""
        keys = [a.key for a in self.actions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate actions")

        ctx = info.context or {}
        snapshot = ctx.get("snapshot")
        evaluation = ctx.get("evaluation")
        if snapshot is None or evaluation is None:
            return self
        extracted: set[int] = ctx.get("extracted_document_ids", set())

        for a in self.actions:
            doc = snapshot.document(a.document_id) if a.document_id is not None else None
            if a.document_id is not None and doc is None:
                raise ValueError(f"unknown document_id {a.document_id}")
            if a.requirement_code is not None and snapshot.requirement(a.requirement_code) is None:
                raise ValueError(f"unknown requirement_code '{a.requirement_code}'")
            ev = evaluation.get(a.requirement_code) if a.requirement_code else None

            if a.kind in (ActionKind.ACCEPT_DOCUMENT, ActionKind.REJECT_DOCUMENT):
                if doc.status != "pending_review":
                    raise ValueError(f"document {doc.id} is already {doc.status}")
            if a.kind == ActionKind.ACCEPT_DOCUMENT:
                if a.document_id not in extracted:
                    raise ValueError(f"document {a.document_id} has no validated extraction")
                if not any(
                    e.status == RequirementStatus.SATISFIED and a.document_id in e.document_ids
                    for e in evaluation.evaluations
                ):
                    raise ValueError(f"document {a.document_id} does not satisfy any requirement")
            if a.kind == ActionKind.MARK_REQUIREMENT_SATISFIED:
                if ev is None or ev.status != RequirementStatus.SATISFIED:
                    raise ValueError(f"'{a.requirement_code}' was not evaluated as satisfied")
                if a.document_id not in ev.document_ids:
                    raise ValueError(f"document {a.document_id} is not evidence for '{a.requirement_code}'")
            if a.kind == ActionKind.CHANGE_CASE_STATUS:
                if any(e.status != RequirementStatus.SATISFIED for e in evaluation.evaluations):
                    raise ValueError("case can only be marked ready when every requirement is satisfied")
        return self
