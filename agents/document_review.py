from typing import Any

from .base import Agent, ToolInvocation
from .schemas import DocumentExtractionResult, DocumentReviewInput
from .tools import GET_DOCUMENT_TEXT, GET_EXTRACTION_SCHEMA, AgentContext


class DocumentReviewAgent(Agent[DocumentReviewInput, DocumentExtractionResult]):
    """Classifies one document and extracts structured fields from it."""

    name = "DocumentReviewAgent"
    output_schema = DocumentExtractionResult
    tools = (GET_DOCUMENT_TEXT, GET_EXTRACTION_SCHEMA)
    system_prompt = """\
You review a single document uploaded to a case file.
1. Call get_document_text to read the document.
2. Decide its document_type. If none fits, use "unknown" and extract no fields.
3. Call get_extraction_schema for that type and extract only those fields.
Rules:
- Every field needs `evidence`: a verbatim quote from the document containing the value.
- Omit a field rather than guess. Record anything unclear in `issues`.
- Dates as YYYY-MM-DD; money as a plain decimal (12500.00); currency as an ISO code.
Respond with a DocumentExtractionResult JSON object only."""

    def validation_context(
        self, task: DocumentReviewInput, ctx: AgentContext, tool_log: list[ToolInvocation]
    ) -> dict[str, Any]:
        doc = ctx.snapshot.document(task.document_id)
        return {"document_id": task.document_id, "document_text": doc.text if doc else ""}
