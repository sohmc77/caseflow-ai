"""Checklist templates per case type (fictional requirements for the demo).

Rules are written as the same Pydantic models the agent layer validates, so a
typo in a template fails loudly at import time and not halfway through a review.
"""

from agents.schemas import (
    DocumentType,
    MinAmountRule,
    MinValidityRule,
    NameMatchesApplicantRule,
    RequiredFieldsRule,
    Rule,
)
from agents.schemas.common import Contract


class RequirementTemplate(Contract):
    code: str
    title: str
    accepted_document_types: list[DocumentType]
    rules: list[Rule] = []


TEMPLATES: dict[str, list[RequirementTemplate]] = {
    "work_permit": [
        RequirementTemplate(
            code="identity",
            title="Passport valid for at least 6 months",
            accepted_document_types=[DocumentType.PASSPORT],
            rules=[
                RequiredFieldsRule(fields=["full_name", "passport_number", "expiry_date"]),
                MinValidityRule(field="expiry_date", days=180),
                NameMatchesApplicantRule(field="full_name"),
            ],
        ),
        RequirementTemplate(
            code="job_offer",
            title="Signed offer of employment",
            accepted_document_types=[DocumentType.EMPLOYMENT_LETTER],
            rules=[
                RequiredFieldsRule(fields=["employer_name", "job_title", "start_date"]),
                NameMatchesApplicantRule(field="employee_name"),
            ],
        ),
        RequirementTemplate(
            code="proof_of_funds",
            title="Bank statement showing at least 10,000 EUR",
            accepted_document_types=[DocumentType.BANK_STATEMENT],
            rules=[
                MinAmountRule(field="closing_balance", amount=10000, currency="EUR"),
                NameMatchesApplicantRule(field="account_holder"),
            ],
        ),
    ],
    "study_permit": [
        RequirementTemplate(
            code="identity",
            title="Passport valid for at least 12 months",
            accepted_document_types=[DocumentType.PASSPORT],
            rules=[
                MinValidityRule(field="expiry_date", days=365),
                NameMatchesApplicantRule(field="full_name"),
            ],
        ),
        RequirementTemplate(
            code="prior_education",
            title="Degree certificate",
            accepted_document_types=[DocumentType.DEGREE_CERTIFICATE],
            rules=[
                RequiredFieldsRule(fields=["degree", "institution"]),
                NameMatchesApplicantRule(field="graduate_name"),
            ],
        ),
        RequirementTemplate(
            code="proof_of_funds",
            title="Bank statement showing at least 8,000 EUR",
            accepted_document_types=[DocumentType.BANK_STATEMENT],
            rules=[MinAmountRule(field="closing_balance", amount=8000, currency="EUR")],
        ),
    ],
}

CASE_TYPE_CHOICES = [(key, key.replace("_", " ").title()) for key in TEMPLATES]
