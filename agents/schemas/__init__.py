"""Pydantic contracts for every agent input, output, and tool.

Nothing produced by an LLM reaches the database without passing through one of
these models.
"""

from .actions import ActionKind, NextStepPlan, ProposedActionDraft
from .case import (
    CaseSnapshot,
    DocumentSnapshot,
    MinAmountRule,
    MinValidityRule,
    NameMatchesApplicantRule,
    RequiredFieldsRule,
    RequirementSnapshot,
    Rule,
)
from .checklist import ChecklistEvaluation, RequirementEvaluation, RequirementStatus, RuleCheckResult
from .common import FIELD_SPECS, Contract, DocumentType, FieldKind
from .extraction import DocumentExtractionResult, ExtractedField
from .pipeline import (
    AgentStep,
    AgentTrace,
    CaseReviewResult,
    ChecklistInput,
    DocumentFailure,
    DocumentReviewInput,
    NextStepInput,
)

__all__ = [
    "FIELD_SPECS",
    "ActionKind",
    "AgentStep",
    "AgentTrace",
    "CaseReviewResult",
    "CaseSnapshot",
    "ChecklistEvaluation",
    "ChecklistInput",
    "Contract",
    "DocumentExtractionResult",
    "DocumentFailure",
    "DocumentReviewInput",
    "DocumentSnapshot",
    "DocumentType",
    "ExtractedField",
    "FieldKind",
    "MinAmountRule",
    "MinValidityRule",
    "NameMatchesApplicantRule",
    "NextStepInput",
    "NextStepPlan",
    "ProposedActionDraft",
    "RequiredFieldsRule",
    "RequirementEvaluation",
    "RequirementSnapshot",
    "RequirementStatus",
    "Rule",
    "RuleCheckResult",
]
