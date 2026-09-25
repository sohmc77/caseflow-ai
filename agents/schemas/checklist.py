"""Output contract of the ChecklistAgent."""

from enum import StrEnum
from typing import Self

from pydantic import Field, ValidationInfo, model_validator

from .common import Contract


class RequirementStatus(StrEnum):
    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"  # a candidate document exists but fails a rule
    MISSING = "missing"  # no candidate document at all
    NEEDS_REVIEW = "needs_review"  # automation could not decide; a human must look


class RuleCheckResult(Contract):
    """Output of the deterministic ``check_requirement_against_checklist`` tool."""

    requirement_code: str
    document_id: int
    passed: bool
    failures: list[str] = []


class RequirementEvaluation(Contract):
    requirement_code: str
    status: RequirementStatus
    document_ids: list[int] = []
    reasons: list[str] = Field(min_length=1)


class ChecklistEvaluation(Contract):
    case_id: int
    evaluations: list[RequirementEvaluation]

    @model_validator(mode="after")
    def _consistent_with_run(self, info: ValidationInfo) -> Self:
        codes = [e.requirement_code for e in self.evaluations]
        if len(codes) != len(set(codes)):
            raise ValueError("each requirement must be evaluated exactly once")

        ctx = info.context or {}
        if (expected_case := ctx.get("case_id")) is not None and self.case_id != expected_case:
            raise ValueError(f"case_id must be {expected_case}")
        if (expected_codes := ctx.get("requirement_codes")) is not None:
            if missing := sorted(set(expected_codes) - set(codes)):
                raise ValueError(f"requirements not evaluated: {missing}")
            if unknown := sorted(set(codes) - set(expected_codes)):
                raise ValueError(f"unknown requirement codes: {unknown}")
        if (known_docs := ctx.get("document_ids")) is not None:
            for e in self.evaluations:
                if unknown := sorted(set(e.document_ids) - set(known_docs)):
                    raise ValueError(f"{e.requirement_code}: unknown document ids {unknown}")

        # The model may only claim SATISFIED when the deterministic rule checker
        # agreed during this run. Judgement is the model's; facts are the tool's.
        if (passing := ctx.get("passing_checks")) is not None:
            for e in self.evaluations:
                if e.status == RequirementStatus.SATISFIED and not any(
                    (e.requirement_code, doc_id) in passing for doc_id in e.document_ids
                ):
                    raise ValueError(
                        f"{e.requirement_code}: 'satisfied' requires a passing "
                        "check_requirement_against_checklist result for a listed document"
                    )
        return self

    def get(self, requirement_code: str) -> RequirementEvaluation | None:
        return next((e for e in self.evaluations if e.requirement_code == requirement_code), None)
