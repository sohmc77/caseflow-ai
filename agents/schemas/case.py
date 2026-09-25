"""Read-only snapshot of a case, handed to agents instead of ORM objects.

Agents never receive a database handle. They see an immutable snapshot built by
the service layer, which is what makes "the AI cannot mutate case state" a
structural property rather than a promise.
"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field

from .common import Contract, DocumentType


class RequiredFieldsRule(Contract):
    type: Literal["required_fields"] = "required_fields"
    fields: list[str] = Field(min_length=1)


class MinValidityRule(Contract):
    """A date field must be at least ``days`` after the case's ``as_of`` date."""

    type: Literal["min_validity_days"] = "min_validity_days"
    field: str
    days: int = Field(gt=0)


class MinAmountRule(Contract):
    type: Literal["min_amount"] = "min_amount"
    field: str
    amount: Decimal = Field(gt=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    currency_field: str = "currency"


class NameMatchesApplicantRule(Contract):
    type: Literal["name_matches_applicant"] = "name_matches_applicant"
    field: str


Rule = Annotated[
    RequiredFieldsRule | MinValidityRule | MinAmountRule | NameMatchesApplicantRule,
    Field(discriminator="type"),
]


class RequirementSnapshot(Contract):
    code: str
    title: str
    accepted_document_types: list[DocumentType] = Field(min_length=1)
    rules: list[Rule] = []
    status: Literal["open", "satisfied"] = "open"


class DocumentSnapshot(Contract):
    id: int
    filename: str
    text: str
    status: Literal["pending_review", "accepted", "rejected"] = "pending_review"


class CaseSnapshot(Contract):
    id: int
    case_type: str
    applicant_name: str
    status: str = "open"
    as_of: date  # pinned "today" so rule checks are reproducible in tests/evals
    documents: list[DocumentSnapshot] = []
    requirements: list[RequirementSnapshot] = []

    def document(self, document_id: int) -> DocumentSnapshot | None:
        return next((d for d in self.documents if d.id == document_id), None)

    def requirement(self, code: str) -> RequirementSnapshot | None:
        return next((r for r in self.requirements if r.code == code), None)
