"""Primitives shared by every agent contract."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Contract(BaseModel):
    """Base class for every agent input/output contract.

    - ``extra="forbid"``: a model that invents keys fails validation instead of
      silently smuggling data through.
    - ``frozen=True``: once validated, a contract is a value, not a mutable record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentType(StrEnum):
    PASSPORT = "passport"
    EMPLOYMENT_LETTER = "employment_letter"
    BANK_STATEMENT = "bank_statement"
    DEGREE_CERTIFICATE = "degree_certificate"
    UNKNOWN = "unknown"


class FieldKind(StrEnum):
    TEXT = "text"
    DATE = "date"  # ISO-8601, YYYY-MM-DD
    MONEY = "money"  # plain decimal string, no symbols or separators
    CURRENCY = "currency"  # ISO-4217 code


# Single source of truth for what can be extracted from each document type.
# Used by the validator (rejects unknown fields) and exposed to the model via
# the ``get_extraction_schema`` tool, so prompt and validation cannot drift apart.
FIELD_SPECS: dict[DocumentType, dict[str, FieldKind]] = {
    DocumentType.PASSPORT: {
        "full_name": FieldKind.TEXT,
        "passport_number": FieldKind.TEXT,
        "nationality": FieldKind.TEXT,
        "date_of_birth": FieldKind.DATE,
        "expiry_date": FieldKind.DATE,
    },
    DocumentType.EMPLOYMENT_LETTER: {
        "employee_name": FieldKind.TEXT,
        "employer_name": FieldKind.TEXT,
        "job_title": FieldKind.TEXT,
        "start_date": FieldKind.DATE,
        "annual_salary": FieldKind.MONEY,
    },
    DocumentType.BANK_STATEMENT: {
        "account_holder": FieldKind.TEXT,
        "statement_date": FieldKind.DATE,
        "closing_balance": FieldKind.MONEY,
        "currency": FieldKind.CURRENCY,
    },
    DocumentType.DEGREE_CERTIFICATE: {
        "graduate_name": FieldKind.TEXT,
        "degree": FieldKind.TEXT,
        "institution": FieldKind.TEXT,
        "award_date": FieldKind.DATE,
    },
    DocumentType.UNKNOWN: {},
}


def normalize_text(value: str) -> str:
    """Case- and whitespace-insensitive form used for grounding checks."""
    return " ".join(value.split()).casefold()
