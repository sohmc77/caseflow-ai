"""Tools agents may call.

Every tool is a pure function of an immutable ``AgentContext``. None of them can
write anything; the only way to change a case is a human approving a
ProposedAction.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel

from . import rules
from .providers.base import ToolSpec
from .schemas import FIELD_SPECS, CaseSnapshot, DocumentExtractionResult, RuleCheckResult
from .schemas.tools import (
    CaseSummary,
    ChecklistItemView,
    ChecklistView,
    CheckRequirementArgs,
    DocumentSummary,
    DocumentText,
    ExtractionSchema,
    GetDocumentTextArgs,
    GetExtractionSchemaArgs,
    NoArgs,
)


@dataclass(frozen=True)
class AgentContext:
    snapshot: CaseSnapshot
    extractions: Mapping[int, DocumentExtractionResult] = field(default_factory=dict)


class ToolError(Exception):
    """A recoverable error, reported back to the model so it can correct itself."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[AgentContext, BaseModel], BaseModel]

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.args_model.model_json_schema())


def _get_document_text(ctx: AgentContext, args: GetDocumentTextArgs) -> DocumentText:
    doc = ctx.snapshot.document(args.document_id)
    if doc is None:
        raise ToolError(f"no document with id {args.document_id} in this case")
    return DocumentText(document_id=doc.id, filename=doc.filename, text=doc.text)


def _get_extraction_schema(ctx: AgentContext, args: GetExtractionSchemaArgs) -> ExtractionSchema:
    return ExtractionSchema(document_type=args.document_type, fields=FIELD_SPECS[args.document_type])


def _list_checklist_items(ctx: AgentContext, args: NoArgs) -> ChecklistView:
    return ChecklistView(
        items=[
            ChecklistItemView(
                code=r.code,
                title=r.title,
                accepted_document_types=r.accepted_document_types,
                status=r.status,
                rules=[rules.describe(rule) for rule in r.rules],
            )
            for r in ctx.snapshot.requirements
        ]
    )


def _check_requirement(ctx: AgentContext, args: CheckRequirementArgs) -> RuleCheckResult:
    requirement = ctx.snapshot.requirement(args.requirement_code)
    if requirement is None:
        raise ToolError(f"unknown requirement_code '{args.requirement_code}'")
    extraction = ctx.extractions.get(args.document_id)
    if extraction is None:
        raise ToolError(f"document {args.document_id} has no validated extraction")
    return rules.check_requirement(requirement, extraction, ctx.snapshot)


def _get_case_summary(ctx: AgentContext, args: NoArgs) -> CaseSummary:
    s = ctx.snapshot
    return CaseSummary(
        case_id=s.id,
        case_type=s.case_type,
        status=s.status,
        documents=[DocumentSummary(id=d.id, filename=d.filename, status=d.status) for d in s.documents],
        requirements={r.code: r.status for r in s.requirements},
    )


GET_DOCUMENT_TEXT = Tool(
    "get_document_text",
    "Return the full text of one document in the case.",
    GetDocumentTextArgs,
    _get_document_text,
)
GET_EXTRACTION_SCHEMA = Tool(
    "get_extraction_schema",
    "Return the fields (and their formats) that may be extracted for a document type.",
    GetExtractionSchemaArgs,
    _get_extraction_schema,
)
LIST_CHECKLIST_ITEMS = Tool(
    "list_checklist_items",
    "List the case's checklist requirements and their rules.",
    NoArgs,
    _list_checklist_items,
)
CHECK_REQUIREMENT = Tool(
    "check_requirement_against_checklist",
    "Deterministically check one document's validated extraction against one requirement's rules. "
    "A requirement may only be reported as satisfied if this tool returned passed=true.",
    CheckRequirementArgs,
    _check_requirement,
)
GET_CASE_SUMMARY = Tool(
    "get_case_summary",
    "Return the case status, documents with their review status, and requirement statuses.",
    NoArgs,
    _get_case_summary,
)
