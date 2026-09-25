"""Output contract of the DocumentReviewAgent."""

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Self

from pydantic import Field, ValidationInfo, field_validator, model_validator

from .common import FIELD_SPECS, Confidence, Contract, DocumentType, FieldKind, normalize_text


class ExtractedField(Contract):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description="Verbatim quote from the document that supports the value.")
    confidence: Confidence


class DocumentExtractionResult(Contract):
    document_id: int
    document_type: DocumentType
    fields: list[ExtractedField] = []
    issues: list[str] = Field(default=[], description="Anything a human reviewer should know.")
    confidence: Confidence

    @field_validator("fields")
    @classmethod
    def _unique_field_names(cls, fields: list[ExtractedField]) -> list[ExtractedField]:
        names = [f.name for f in fields]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate fields: {duplicates}")
        return fields

    @model_validator(mode="after")
    def _fields_match_spec(self) -> Self:
        """Field names and value formats must match FIELD_SPECS for the document type."""
        spec = FIELD_SPECS[self.document_type]
        for f in self.fields:
            kind = spec.get(f.name)
            if kind is None:
                raise ValueError(f"field '{f.name}' is not defined for document type '{self.document_type}'")
            _check_format(f, kind)
        return self

    @model_validator(mode="after")
    def _grounded_in_source(self, info: ValidationInfo) -> Self:
        """Anti-hallucination check; only runs when the caller supplies the source.

        Every ``evidence`` string must be a verbatim quote from the document, and
        text values must appear inside their evidence.
        """
        ctx = info.context or {}
        expected_id = ctx.get("document_id")
        if expected_id is not None and self.document_id != expected_id:
            raise ValueError(f"document_id must be {expected_id}, got {self.document_id}")
        source = ctx.get("document_text")
        if source is None:
            return self
        haystack = normalize_text(source)
        for f in self.fields:
            if normalize_text(f.evidence) not in haystack:
                raise ValueError(f"evidence for '{f.name}' is not a verbatim quote from the document")
            kind = FIELD_SPECS[self.document_type][f.name]
            if kind == FieldKind.TEXT and normalize_text(f.value) not in normalize_text(f.evidence):
                raise ValueError(f"value for '{f.name}' does not appear in its evidence")
        return self

    def value(self, field_name: str) -> str | None:
        return next((f.value for f in self.fields if f.name == field_name), None)


def _check_format(f: ExtractedField, kind: FieldKind) -> None:
    if kind == FieldKind.DATE:
        try:
            date.fromisoformat(f.value)
        except ValueError:
            raise ValueError(f"'{f.name}' must be an ISO date (YYYY-MM-DD), got '{f.value}'") from None
    elif kind == FieldKind.MONEY:
        try:
            Decimal(f.value)
        except InvalidOperation:
            raise ValueError(f"'{f.name}' must be a plain decimal like 12500.00, got '{f.value}'") from None
    elif kind == FieldKind.CURRENCY:
        if not (len(f.value) == 3 and f.value.isalpha() and f.value.isupper()):
            raise ValueError(f"'{f.name}' must be an ISO-4217 code like EUR, got '{f.value}'")
