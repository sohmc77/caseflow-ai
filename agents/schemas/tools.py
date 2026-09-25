"""Argument and result contracts for the tools agents can call.

Tool arguments come from the model, so they are validated exactly like final
outputs. The JSON Schema of each ``*Args`` model is what the provider sends to
the LLM as the tool's parameter definition.
"""

from typing import Literal

from pydantic import Field

from .common import Contract, DocumentType, FieldKind


class NoArgs(Contract):
    pass


class GetDocumentTextArgs(Contract):
    document_id: int


class DocumentText(Contract):
    document_id: int
    filename: str
    text: str


class GetExtractionSchemaArgs(Contract):
    document_type: DocumentType


class ExtractionSchema(Contract):
    document_type: DocumentType
    fields: dict[str, FieldKind]


class ChecklistItemView(Contract):
    code: str
    title: str
    accepted_document_types: list[DocumentType]
    status: Literal["open", "satisfied"]
    rules: list[str] = Field(description="Human-readable rule descriptions.")


class ChecklistView(Contract):
    items: list[ChecklistItemView]


class CheckRequirementArgs(Contract):
    requirement_code: str
    document_id: int


class DocumentSummary(Contract):
    id: int
    filename: str
    status: str


class CaseSummary(Contract):
    case_id: int
    case_type: str
    status: str
    documents: list[DocumentSummary]
    requirements: dict[str, Literal["open", "satisfied"]]
