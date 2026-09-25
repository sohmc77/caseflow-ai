"""Agent loop behaviour, driven by scripted mock responses (no network)."""

import json

import pytest

from agents.base import AgentError
from agents.checklist import ChecklistAgent
from agents.document_review import DocumentReviewAgent
from agents.providers import CompletionResponse, MockProvider, ToolCall
from agents.schemas import ChecklistInput, DocumentReviewInput
from agents.tools import AgentContext

from .conftest import make_extraction

VALID = json.dumps(
    {
        "document_id": 10,
        "document_type": "passport",
        "fields": [
            {"name": "passport_number", "value": "AR4471902", "evidence": "Passport No: AR4471902", "confidence": 0.95}
        ],
        "confidence": 0.9,
    }
)


def tool_call(name: str, args: dict | str, call_id: str = "c1") -> CompletionResponse:
    raw = args if isinstance(args, str) else json.dumps(args)
    return CompletionResponse(tool_calls=[ToolCall(id=call_id, name=name, arguments_json=raw)], model="scripted")


def run_review(snapshot, script):
    provider = MockProvider(script=script)
    result = DocumentReviewAgent(provider).run(DocumentReviewInput(document_id=10), AgentContext(snapshot))
    return result, provider


def test_retries_after_malformed_json_then_succeeds(snapshot):
    result, provider = run_review(snapshot, ["{not json", VALID])

    assert result.output.value("passport_number") == "AR4471902"
    assert result.trace.validation_retries == 1
    correction = provider.requests[1].messages[-1]
    assert correction.role == "user" and "failed validation" in correction.content


def test_hallucinated_evidence_is_rejected_and_corrected(snapshot):
    invented = VALID.replace("AR4471902", "ZZ0000000")
    result, provider = run_review(snapshot, [invented, VALID])

    assert result.trace.validation_retries == 1
    assert "not a verbatim quote" in provider.requests[1].messages[-1].content


def test_gives_up_after_max_retries_with_trace(snapshot):
    with pytest.raises(AgentError, match="failed validation after 3 attempts") as exc_info:
        run_review(snapshot, ["{}", "{}", "{}"])
    trace = exc_info.value.trace
    assert [s.type for s in trace.steps] == ["validation_error"] * 3


def test_tool_errors_are_returned_to_the_model_not_raised(snapshot):
    script = [
        tool_call("get_document_text", {"document_id": "not-an-int"}),
        tool_call("get_document_text", {"document_id": 999}),
        tool_call("delete_case", {}),
        tool_call("get_document_text", {"document_id": 10}),
        VALID,
    ]
    result, provider = run_review(snapshot, script)

    tool_messages = [m for m in provider.requests[-1].messages if m.role == "tool"]
    errors = [json.loads(m.content).get("error", "") for m in tool_messages[:3]]
    assert "invalid arguments" in errors[0]
    assert "no document with id 999" in errors[1]
    assert "unknown tool 'delete_case'" in errors[2]
    assert "Passport No" in json.loads(tool_messages[3].content)["text"]
    assert [s.type for s in result.trace.steps] == ["tool_error"] * 3 + ["tool_call", "output"]


def test_stops_after_max_turns(snapshot):
    endless = [tool_call("get_document_text", {"document_id": 10}, f"c{i}") for i in range(20)]
    with pytest.raises(AgentError, match="no final answer within 8 turns"):
        run_review(snapshot, endless)


def test_checklist_agent_cannot_claim_satisfied_without_tool_check(snapshot):
    passport = make_extraction(
        10, "passport", full_name="Amara Okafor", passport_number="AR4471902", expiry_date="2031-03-14"
    )
    claim = json.dumps(
        {
            "case_id": 1,
            "evaluations": [
                {"requirement_code": "identity", "status": "satisfied", "document_ids": [10], "reasons": ["trust me"]},
                {"requirement_code": "job_offer", "status": "missing", "reasons": ["none"]},
                {"requirement_code": "proof_of_funds", "status": "missing", "reasons": ["none"]},
            ],
        }
    )
    check = tool_call("check_requirement_against_checklist", {"requirement_code": "identity", "document_id": 10})
    provider = MockProvider(script=[claim, check, claim])  # skips the tool, gets rejected, then complies
    ctx = AgentContext(snapshot, {10: passport})
    result = ChecklistAgent(provider).run(ChecklistInput(case_id=1, extractions=[passport]), ctx)

    assert result.trace.validation_retries == 1
    assert "requires a passing check_requirement_against_checklist" in provider.requests[1].messages[-1].content
    assert result.output.get("identity").status == "satisfied"
