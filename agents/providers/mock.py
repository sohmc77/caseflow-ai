"""Deterministic, offline stand-in for an LLM.

It goes through the same protocol as a real model: it asks for tools, reads
their results from the conversation, and returns JSON that the agent then
validates. The "intelligence" is simple heuristics (label regexes, keyword
classification), so it makes realistic mistakes on prose-style documents. The
eval harness shows those misses; it does not hide them.

Two modes:
- default: heuristic simulation, dispatched on the requested output schema.
- ``script=[...]``: return pre-baked responses in order (used by unit tests to
  inject malformed JSON, bad tool calls, etc.).
"""

import json
import re
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal, InvalidOperation
from itertools import count

from ..schemas import (
    ActionKind,
    ChecklistInput,
    DocumentReviewInput,
    DocumentType,
    FieldKind,
    NextStepInput,
    RequirementStatus,
)
from .base import CompletionRequest, CompletionResponse, LLMProvider, Message, ToolCall

MODEL_NAME = "mock-heuristic-v1"


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, script: list[CompletionResponse | str] | None = None):
        self._script = list(script) if script is not None else None
        self._ids = count(1)
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if self._script is not None:
            if not self._script:
                raise AssertionError("MockProvider script exhausted")
            item = self._script.pop(0)
            return item if isinstance(item, CompletionResponse) else _final(item)
        handler = _HANDLERS.get(request.output_schema_name)
        if handler is None:
            raise ValueError(f"MockProvider has no simulation for {request.output_schema_name}")
        return handler(self, _Conversation(request.messages))

    def call(self, *calls: tuple[str, dict]) -> CompletionResponse:
        return CompletionResponse(
            tool_calls=[
                ToolCall(id=f"call_{next(self._ids)}", name=name, arguments_json=json.dumps(args))
                for name, args in calls
            ],
            model=MODEL_NAME,
        )


def _final(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, model=MODEL_NAME)


class _Conversation:
    """Read-only view of the messages so far: the task input and tool results."""

    def __init__(self, messages: list[Message]):
        self.task_json = next(m.content for m in messages if m.role == "user")
        self.results: dict[str, list[dict]] = {}
        for m in messages:
            if m.role == "tool" and m.name:
                self.results.setdefault(m.name, []).append(json.loads(m.content or "{}"))

    def has(self, tool: str) -> bool:
        return tool in self.results

    def last(self, tool: str) -> dict:
        return self.results[tool][-1]


# ---------------------------------------------------------------- document review

_CLASSIFIER_KEYWORDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.PASSPORT: ("passport", "nationality", "place of birth"),
    DocumentType.EMPLOYMENT_LETTER: ("offer of employment", "employment", "employer", "job title"),
    DocumentType.BANK_STATEMENT: ("statement", "closing balance", "account"),
    DocumentType.DEGREE_CERTIFICATE: ("degree", "conferred", "university", "graduate"),
}

_LABELS: dict[str, tuple[str, ...]] = {
    "full_name": ("full name", "name", "surname and given names"),
    "passport_number": ("passport no", "passport number", "document no"),
    "nationality": ("nationality",),
    "date_of_birth": ("date of birth", "dob"),
    "expiry_date": ("date of expiry", "expiry date", "expires"),
    "employee_name": ("employee", "employee name"),
    "employer_name": ("employer", "employer name", "company"),
    "job_title": ("job title", "position"),
    "start_date": ("start date", "commencement date"),
    "annual_salary": ("annual salary", "gross annual salary"),
    "account_holder": ("account holder", "account name"),
    "statement_date": ("statement date",),
    "closing_balance": ("closing balance",),
    "currency": ("currency",),
    "graduate_name": ("graduate", "graduate name"),
    "degree": ("degree", "award"),
    "institution": ("institution", "university"),
    "award_date": ("date of award", "award date", "conferred on"),
}

_DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%d/%m/%Y", "%d.%m.%Y", "%B %d, %Y")


def _classify(text: str) -> DocumentType:
    lowered = text.casefold()
    scores = {t: sum(lowered.count(k) for k in kws) for t, kws in _CLASSIFIER_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else DocumentType.UNKNOWN


def _find_labelled(field: str, text: str) -> tuple[str, str] | None:
    """Return (raw value, evidence line) for a ``Label: value`` line."""
    for label in _LABELS.get(field, ()):
        pattern = rf"^[ \t]*{re.escape(label)}[ \t]*[:\-][ \t]*(?P<value>\S.*?)[ \t]*$"
        if match := re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return match.group("value"), match.group(0).strip()
    return None


def _normalize(raw: str, kind: FieldKind) -> str | None:
    if kind == FieldKind.DATE:
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(raw.strip(), fmt).date().isoformat()
            except ValueError:
                continue
        return None
    if kind == FieldKind.MONEY:
        if match := re.search(r"\d[\d,]*(?:\.\d+)?", raw):
            try:
                return str(Decimal(match.group(0).replace(",", "")))
            except InvalidOperation:
                return None
        return None
    if kind == FieldKind.CURRENCY:
        match = re.search(r"\b[A-Z]{3}\b", raw.upper())
        return match.group(0) if match else None
    return raw.strip()


def _simulate_extraction(mock: MockProvider, convo: _Conversation) -> CompletionResponse:
    task = DocumentReviewInput.model_validate_json(convo.task_json)
    if not convo.has("get_document_text"):
        return mock.call(("get_document_text", {"document_id": task.document_id}))
    text = convo.last("get_document_text").get("text", "")
    doc_type = _classify(text)
    if doc_type != DocumentType.UNKNOWN and not convo.has("get_extraction_schema"):
        return mock.call(("get_extraction_schema", {"document_type": doc_type}))

    spec = convo.last("get_extraction_schema")["fields"] if doc_type != DocumentType.UNKNOWN else {}
    fields, issues = [], []
    for name, kind in spec.items():
        found = _find_labelled(name, text)
        value = _normalize(found[0], FieldKind(kind)) if found else None
        if value is None:
            issues.append(f"could not locate '{name}'")
            continue
        fields.append({"name": name, "value": value, "evidence": found[1], "confidence": 0.9})
    if doc_type == DocumentType.UNKNOWN:
        issues.append("document type not recognised")
    confidence = round(len(fields) / len(spec), 2) if spec else 0.0
    return _final(
        json.dumps(
            {
                "document_id": task.document_id,
                "document_type": doc_type,
                "fields": fields,
                "issues": issues,
                "confidence": confidence,
            }
        )
    )


# ---------------------------------------------------------------- checklist


def _simulate_checklist(mock: MockProvider, convo: _Conversation) -> CompletionResponse:
    task = ChecklistInput.model_validate_json(convo.task_json)
    if not convo.has("list_checklist_items"):
        return mock.call(("list_checklist_items", {}))
    items = convo.last("list_checklist_items")["items"]

    candidates = {
        item["code"]: [e for e in task.extractions if e.document_type in item["accepted_document_types"]]
        for item in items
    }
    if not convo.has("check_requirement_against_checklist"):
        calls = [
            ("check_requirement_against_checklist", {"requirement_code": code, "document_id": e.document_id})
            for code, docs in candidates.items()
            for e in docs
        ]
        if calls:
            return mock.call(*calls)
    checks = convo.results.get("check_requirement_against_checklist", [])

    evaluations = []
    for item in items:
        code = item["code"]
        mine = [c for c in checks if c.get("requirement_code") == code]
        passing = [c["document_id"] for c in mine if c["passed"]]
        if passing:
            evaluations.append(
                {
                    "requirement_code": code,
                    "status": RequirementStatus.SATISFIED,
                    "document_ids": passing,
                    "reasons": ["all rules passed"],
                }
            )
        elif mine:
            evaluations.append(
                {
                    "requirement_code": code,
                    "status": RequirementStatus.NOT_SATISFIED,
                    "document_ids": [c["document_id"] for c in mine],
                    "reasons": [f for c in mine for f in c["failures"]] or ["rule check failed"],
                }
            )
        elif task.unreviewed_document_ids:
            evaluations.append(
                {
                    "requirement_code": code,
                    "status": RequirementStatus.NEEDS_REVIEW,
                    "document_ids": [],
                    "reasons": ["no matching document, but some documents could not be reviewed automatically"],
                }
            )
        else:
            evaluations.append(
                {
                    "requirement_code": code,
                    "status": RequirementStatus.MISSING,
                    "document_ids": [],
                    "reasons": [f"no document of type {item['accepted_document_types']} provided"],
                }
            )
    return _final(json.dumps({"case_id": task.case_id, "evaluations": evaluations}))


# ---------------------------------------------------------------- next steps


def _simulate_next_steps(mock: MockProvider, convo: _Conversation) -> CompletionResponse:
    task = NextStepInput.model_validate_json(convo.task_json)
    if not convo.has("get_case_summary"):
        return mock.call(("get_case_summary", {}))
    summary = convo.last("get_case_summary")
    pending = {d["id"] for d in summary["documents"] if d["status"] == "pending_review"}
    req_status = summary["requirements"]

    actions: list[dict] = []
    seen: set[tuple] = set()

    def add(**action) -> None:
        key = (action["kind"], action.get("document_id"), action.get("requirement_code"))
        if key not in seen:
            seen.add(key)
            actions.append(action)

    for ev in task.evaluation.evaluations:
        code = ev.requirement_code
        if ev.status == RequirementStatus.SATISFIED:
            for doc_id in ev.document_ids:
                if doc_id in pending:
                    add(
                        kind=ActionKind.ACCEPT_DOCUMENT,
                        document_id=doc_id,
                        rationale=f"Document {doc_id} passed every rule for '{code}'.",
                    )
            if req_status.get(code) == "open":
                add(
                    kind=ActionKind.MARK_REQUIREMENT_SATISFIED,
                    requirement_code=code,
                    document_id=ev.document_ids[0],
                    rationale=f"Rule checks passed for '{code}'.",
                )
        elif ev.status == RequirementStatus.NOT_SATISFIED:
            reasons = "; ".join(ev.reasons)
            for doc_id in ev.document_ids:
                if doc_id in pending:
                    add(kind=ActionKind.REJECT_DOCUMENT, document_id=doc_id, rationale=f"Fails '{code}': {reasons}")
            add(
                kind=ActionKind.REQUEST_DOCUMENT,
                requirement_code=code,
                message=f"Please provide a replacement document for '{code}'. Issue: {reasons}.",
                rationale=f"Existing document does not meet '{code}': {reasons}",
            )
        elif ev.status == RequirementStatus.MISSING:
            add(
                kind=ActionKind.REQUEST_DOCUMENT,
                requirement_code=code,
                message=f"Please upload the document required for '{code}'.",
                rationale=f"No document has been provided for '{code}'.",
            )
        else:
            add(
                kind=ActionKind.FLAG_FOR_REVIEW,
                requirement_code=code,
                rationale=f"Automation could not decide '{code}': {'; '.join(ev.reasons)}",
            )
    for doc_id in task.unreviewed_document_ids:
        add(
            kind=ActionKind.FLAG_FOR_REVIEW,
            document_id=doc_id,
            rationale=f"Document {doc_id} could not be reviewed automatically.",
        )

    all_ok = all(ev.status == RequirementStatus.SATISFIED for ev in task.evaluation.evaluations)
    if all_ok and summary["status"] != "ready_for_submission":
        add(
            kind=ActionKind.CHANGE_CASE_STATUS,
            new_case_status="ready_for_submission",
            rationale="Every checklist requirement is satisfied.",
        )

    open_count = sum(ev.status != RequirementStatus.SATISFIED for ev in task.evaluation.evaluations)
    text = "All requirements satisfied." if all_ok else f"{open_count} requirement(s) need attention."
    return _final(json.dumps({"summary": text, "actions": actions}, default=str))


_HANDLERS: dict[str, Callable[[MockProvider, _Conversation], CompletionResponse]] = {
    "DocumentExtractionResult": _simulate_extraction,
    "ChecklistEvaluation": _simulate_checklist,
    "NextStepPlan": _simulate_next_steps,
}
