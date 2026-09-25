"""The contracts are the safety boundary, so each invariant is tested directly."""

import json

import pytest
from pydantic import ValidationError

from agents.schemas import (
    ActionKind,
    ChecklistEvaluation,
    DocumentExtractionResult,
    NextStepPlan,
    ProposedActionDraft,
)

from .conftest import PASSPORT_TEXT


def extraction_json(**overrides) -> str:
    data = {
        "document_id": 10,
        "document_type": "passport",
        "fields": [
            {"name": "passport_number", "value": "AR4471902", "evidence": "Passport No: AR4471902", "confidence": 0.9}
        ],
        "confidence": 0.9,
    }
    data.update(overrides)
    return json.dumps(data)


class TestDocumentExtractionResult:
    def test_valid_and_grounded(self):
        result = DocumentExtractionResult.model_validate_json(
            extraction_json(), context={"document_id": 10, "document_text": PASSPORT_TEXT}
        )
        assert result.value("passport_number") == "AR4471902"

    def test_rejects_field_not_defined_for_type(self):
        bad = [{"name": "annual_salary", "value": "1", "evidence": "x", "confidence": 1}]
        with pytest.raises(ValidationError, match="not defined for document type"):
            DocumentExtractionResult.model_validate_json(extraction_json(fields=bad))

    def test_rejects_non_iso_date(self):
        bad = [{"name": "expiry_date", "value": "14 MAR 2031", "evidence": "x", "confidence": 1}]
        with pytest.raises(ValidationError, match="ISO date"):
            DocumentExtractionResult.model_validate_json(extraction_json(fields=bad))

    def test_rejects_extra_keys(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            DocumentExtractionResult.model_validate_json(extraction_json(approved=True))

    def test_rejects_evidence_not_in_source(self):
        invented = [{"name": "passport_number", "value": "ZZ999", "evidence": "Passport No: ZZ999", "confidence": 1}]
        with pytest.raises(ValidationError, match="not a verbatim quote"):
            DocumentExtractionResult.model_validate_json(
                extraction_json(fields=invented), context={"document_text": PASSPORT_TEXT}
            )

    def test_rejects_text_value_not_in_its_evidence(self):
        mismatched = [
            {"name": "nationality", "value": "Velorian", "evidence": "Nationality: Arcadian", "confidence": 1}
        ]
        with pytest.raises(ValidationError, match="does not appear in its evidence"):
            DocumentExtractionResult.model_validate_json(
                extraction_json(fields=mismatched), context={"document_text": PASSPORT_TEXT}
            )

    def test_rejects_wrong_document_id(self):
        with pytest.raises(ValidationError, match="document_id must be 99"):
            DocumentExtractionResult.model_validate_json(extraction_json(), context={"document_id": 99})


class TestChecklistEvaluation:
    ctx = {"case_id": 1, "requirement_codes": ["identity"], "document_ids": [10], "passing_checks": set()}

    def evaluation(self, status: str) -> dict:
        return {
            "case_id": 1,
            "evaluations": [{"requirement_code": "identity", "status": status, "document_ids": [10], "reasons": ["r"]}],
        }

    def test_satisfied_requires_a_passing_tool_check(self):
        with pytest.raises(ValidationError, match="requires a passing"):
            ChecklistEvaluation.model_validate(self.evaluation("satisfied"), context=self.ctx)
        ok_ctx = {**self.ctx, "passing_checks": {("identity", 10)}}
        ChecklistEvaluation.model_validate(self.evaluation("satisfied"), context=ok_ctx)

    def test_every_requirement_must_be_evaluated(self):
        ctx = {**self.ctx, "requirement_codes": ["identity", "proof_of_funds"]}
        with pytest.raises(ValidationError, match="not evaluated"):
            ChecklistEvaluation.model_validate(self.evaluation("missing"), context=ctx)


class TestProposedActionDraft:
    @pytest.mark.parametrize(
        ("payload", "error"),
        [
            ({"kind": "request_document", "requirement_code": "identity"}, "requires \\['message'\\]"),
            ({"kind": "accept_document", "document_id": 1, "requirement_code": "identity"}, "must not set"),
            ({"kind": "change_case_status"}, "requires \\['new_case_status'\\]"),
            ({"kind": "flag_for_review"}, "needs a document_id or requirement_code"),
        ],
    )
    def test_targets_must_match_kind(self, payload, error):
        with pytest.raises(ValidationError, match=error):
            ProposedActionDraft(rationale="a sufficiently long rationale", **payload)


class TestNextStepPlan:
    def test_cannot_mark_ready_unless_everything_is_satisfied(self, snapshot):
        evaluation = ChecklistEvaluation.model_validate(
            {
                "case_id": 1,
                "evaluations": [
                    {"requirement_code": r.code, "status": "missing", "reasons": ["none"]}
                    for r in snapshot.requirements
                ],
            }
        )
        plan = {
            "summary": "s",
            "actions": [
                {
                    "kind": ActionKind.CHANGE_CASE_STATUS,
                    "new_case_status": "ready_for_submission",
                    "rationale": "looks good to me",
                }
            ],
        }
        with pytest.raises(ValidationError, match="every requirement is satisfied"):
            NextStepPlan.model_validate(plan, context={"snapshot": snapshot, "evaluation": evaluation})

    def test_cannot_accept_document_without_extraction(self, snapshot):
        evaluation = ChecklistEvaluation.model_validate(
            {
                "case_id": 1,
                "evaluations": [
                    {"requirement_code": "identity", "status": "satisfied", "document_ids": [10], "reasons": ["ok"]}
                ],
            }
        )
        plan = {"summary": "s", "actions": [{"kind": "accept_document", "document_id": 10, "rationale": "passes all"}]}
        ctx = {"snapshot": snapshot, "evaluation": evaluation, "extracted_document_ids": set()}
        with pytest.raises(ValidationError, match="no validated extraction"):
            NextStepPlan.model_validate(plan, context=ctx)
        NextStepPlan.model_validate(plan, context={**ctx, "extracted_document_ids": {10}})
