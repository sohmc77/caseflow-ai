"""Pydantic models for labelled eval cases.

Fixtures are validated with the same contracts the agents use, so an expected
result that could never be a valid agent output (an unknown field, a bad date,
an unknown requirement) fails when the dataset loads, not during scoring.
"""

from datetime import date
from decimal import Decimal
from typing import Self

from pydantic import Field, model_validator

from agents.schemas import (
    FIELD_SPECS,
    ActionKind,
    CaseSnapshot,
    ChecklistEvaluation,
    Contract,
    DocumentExtractionResult,
    DocumentSnapshot,
    DocumentType,
    ExtractedField,
    FieldKind,
    ProposedActionDraft,
    RequirementSnapshot,
    RequirementStatus,
)
from cases.checklists import TEMPLATES


class ExpectedExtraction(Contract):
    document_type: DocumentType
    fields: dict[str, str] = {}

    @model_validator(mode="after")
    def _fields_exist(self) -> Self:
        if unknown := set(self.fields) - set(FIELD_SPECS[self.document_type]):
            raise ValueError(f"fields not defined for {self.document_type}: {sorted(unknown)}")
        return self


class DocumentReviewCase(Contract):
    id: str
    description: str
    document_lines: list[str]
    expected: ExpectedExtraction

    @property
    def text(self) -> str:
        return "\n".join(self.document_lines)


class FixtureExtraction(Contract):
    """Compact form of an already-validated extraction, used as input to later steps."""

    document_id: int
    document_type: DocumentType
    fields: dict[str, str] = {}

    def to_result(self) -> DocumentExtractionResult:
        return DocumentExtractionResult(
            document_id=self.document_id,
            document_type=self.document_type,
            fields=[
                ExtractedField(name=k, value=v, evidence=f"{k}: {v}", confidence=1.0) for k, v in self.fields.items()
            ],
            confidence=1.0,
        )


class CaseFixture(Contract):
    """Just enough of a case to build a CaseSnapshot from a real checklist template."""

    case_type: str
    applicant_name: str
    as_of: date
    case_status: str = "open"
    extractions: list[FixtureExtraction] = []
    unreviewed_document_ids: list[int] = []
    document_statuses: dict[int, str] = {}
    satisfied_requirements: list[str] = []

    @model_validator(mode="after")
    def _known_case_type(self) -> Self:
        if self.case_type not in TEMPLATES:
            raise ValueError(f"unknown case_type '{self.case_type}'")
        return self

    def snapshot(self) -> CaseSnapshot:
        doc_ids = sorted({e.document_id for e in self.extractions} | set(self.unreviewed_document_ids))
        return CaseSnapshot(
            id=1,
            case_type=self.case_type,
            applicant_name=self.applicant_name,
            status=self.case_status,
            as_of=self.as_of,
            documents=[
                DocumentSnapshot(
                    id=i, filename=f"doc{i}.txt", text="", status=self.document_statuses.get(i, "pending_review")
                )
                for i in doc_ids
            ],
            requirements=[
                RequirementSnapshot(
                    **t.model_dump(),
                    status="satisfied" if t.code in self.satisfied_requirements else "open",
                )
                for t in TEMPLATES[self.case_type]
            ],
        )

    def extraction_results(self) -> dict[int, DocumentExtractionResult]:
        return {e.document_id: e.to_result() for e in self.extractions}


class ChecklistCase(Contract):
    id: str
    description: str
    case: CaseFixture
    expected: dict[str, RequirementStatus]

    @model_validator(mode="after")
    def _covers_every_requirement(self) -> Self:
        codes = {t.code for t in TEMPLATES[self.case.case_type]}
        if set(self.expected) != codes:
            raise ValueError(f"expected must label exactly {sorted(codes)}")
        return self


class ActionMatcher(Contract):
    kind: ActionKind
    document_id: int | None = None
    requirement_code: str | None = None

    def matches(self, draft: ProposedActionDraft) -> bool:
        return (
            draft.kind == self.kind
            and (self.document_id is None or draft.document_id == self.document_id)
            and (self.requirement_code is None or draft.requirement_code == self.requirement_code)
        )

    def __str__(self) -> str:
        target = self.requirement_code or (f"doc {self.document_id}" if self.document_id else "")
        return f"{self.kind}({target})"


class NextStepCase(Contract):
    """Plans are open-ended, so they are scored on must/must-not properties rather than exact match."""

    id: str
    description: str
    case: CaseFixture
    evaluation: ChecklistEvaluation
    must_include: list[ActionMatcher] = Field(min_length=1)
    must_not_include: list[ActionMatcher] = []


def values_equal(document_type: DocumentType, field: str, got: str, want: str) -> bool:
    kind = FIELD_SPECS[document_type][field]
    if kind == FieldKind.MONEY:
        try:
            return Decimal(got) == Decimal(want)
        except ArithmeticError:
            return False
    return " ".join(got.split()).casefold() == " ".join(want.split()).casefold()
