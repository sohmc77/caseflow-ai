from typing import Any

from .base import Agent, ToolInvocation
from .schemas import NextStepInput, NextStepPlan
from .tools import GET_CASE_SUMMARY, AgentContext


class NextStepAgent(Agent[NextStepInput, NextStepPlan]):
    """Turns a checklist evaluation into concrete actions for a human to approve."""

    name = "NextStepAgent"
    output_schema = NextStepPlan
    tools = (GET_CASE_SUMMARY,)
    system_prompt = """\
You propose next actions for a case worker. You cannot change anything yourself;
every action you propose will be approved or rejected by a human.
Call get_case_summary first. Then, based on the checklist evaluation:
- satisfied: accept_document for pending documents that satisfy it, and
  mark_requirement_satisfied if the requirement is still open
- not_satisfied: reject_document for the failing pending documents, and
  request_document with a short, polite message to the applicant
- missing: request_document
- needs_review, or documents that could not be reviewed: flag_for_review
- if every requirement is satisfied: change_case_status to ready_for_submission
Each action needs a specific rationale. Do not propose duplicates.
Respond with a NextStepPlan JSON object only."""

    def validation_context(
        self, task: NextStepInput, ctx: AgentContext, tool_log: list[ToolInvocation]
    ) -> dict[str, Any]:
        return {
            "snapshot": ctx.snapshot,
            "evaluation": task.evaluation,
            "extracted_document_ids": set(ctx.extractions),
        }
