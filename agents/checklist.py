from typing import Any

from .base import Agent, ToolInvocation
from .schemas import ChecklistEvaluation, ChecklistInput, RuleCheckResult
from .tools import CHECK_REQUIREMENT, LIST_CHECKLIST_ITEMS, AgentContext


class ChecklistAgent(Agent[ChecklistInput, ChecklistEvaluation]):
    """Maps validated extractions onto checklist requirements.

    The model decides which documents are relevant to which requirement. Whether
    a document actually passes is decided by the deterministic rule tool, and
    the output validator enforces that.
    """

    name = "ChecklistAgent"
    output_schema = ChecklistEvaluation
    tools = (LIST_CHECKLIST_ITEMS, CHECK_REQUIREMENT)
    system_prompt = """\
You check a case's documents against its requirement checklist.
1. Call list_checklist_items.
2. For each requirement, call check_requirement_against_checklist for every
   extracted document whose type the requirement accepts.
3. Evaluate every requirement exactly once:
   - satisfied: at least one check passed (list only passing document ids)
   - not_satisfied: candidate documents exist but every check failed
   - missing: no candidate document exists
   - needs_review: no candidate, but some documents could not be reviewed
Give concrete reasons, citing rule failures verbatim.
Respond with a ChecklistEvaluation JSON object only."""

    def validation_context(
        self, task: ChecklistInput, ctx: AgentContext, tool_log: list[ToolInvocation]
    ) -> dict[str, Any]:
        passing = {
            (inv.result.requirement_code, inv.result.document_id)
            for inv in tool_log
            if isinstance(inv.result, RuleCheckResult) and inv.result.passed
        }
        return {
            "case_id": task.case_id,
            "requirement_codes": [r.code for r in ctx.snapshot.requirements],
            "document_ids": [d.id for d in ctx.snapshot.documents],
            "passing_checks": passing,
        }
