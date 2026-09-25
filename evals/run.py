"""Run each agent against labelled cases and score the results.

    python -m evals.run                         # mock provider, all suites
    python -m evals.run --provider openai -v    # same cases, real model
    python -m evals.run --json report.json --min-pass-rate 0.8

Exit code 1 if the overall pass rate is below --min-pass-rate (for use as a CI gate).
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import TypeAdapter

from agents.base import AgentError, AgentResult
from agents.checklist import ChecklistAgent
from agents.document_review import DocumentReviewAgent
from agents.next_step import NextStepAgent
from agents.providers import LLMProvider, get_provider
from agents.schemas import (
    CaseSnapshot,
    ChecklistInput,
    DocumentReviewInput,
    DocumentSnapshot,
    NextStepInput,
)
from agents.tools import AgentContext

from .schemas import ChecklistCase, DocumentReviewCase, NextStepCase, values_equal

DATASETS = Path(__file__).parent / "datasets"


@dataclass
class Outcome:
    case_id: str
    passed: bool
    details: list[str] = field(default_factory=list)
    error: str | None = None
    turns: int = 0
    validation_retries: int = 0


@dataclass
class SuiteReport:
    suite: str
    outcomes: list[Outcome]
    metrics: dict[str, float | int] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(o.passed for o in self.outcomes)


def _load(name: str, model):
    return TypeAdapter(list[model]).validate_json((DATASETS / name).read_bytes())


def _run_agent(agent, task, ctx, outcome: Outcome) -> AgentResult | None:
    try:
        result = agent.run(task, ctx)
    except AgentError as exc:
        outcome.error = str(exc)
        outcome.turns, outcome.validation_retries = exc.trace.turns, exc.trace.validation_retries
        return None
    outcome.turns, outcome.validation_retries = result.trace.turns, result.trace.validation_retries
    return result


# ------------------------------------------------------------------ suites


def run_document_review(make_provider: Callable[[], LLMProvider]) -> SuiteReport:
    outcomes, expected_n, extracted_n, correct_n, type_ok = [], 0, 0, 0, 0
    for case in _load("document_review.json", DocumentReviewCase):
        snapshot = CaseSnapshot(
            id=1,
            case_type="eval",
            applicant_name="-",
            as_of="2026-09-01",
            documents=[DocumentSnapshot(id=1, filename=f"{case.id}.txt", text=case.text)],
        )
        outcome = Outcome(case.id, passed=False)
        outcomes.append(outcome)
        expected_n += len(case.expected.fields)
        result = _run_agent(
            DocumentReviewAgent(make_provider()), DocumentReviewInput(document_id=1), AgentContext(snapshot), outcome
        )
        if result is None:
            continue
        got = result.output
        extracted_n += len(got.fields)
        if got.document_type == case.expected.document_type:
            type_ok += 1
        else:
            outcome.details.append(f"type: got {got.document_type}, want {case.expected.document_type}")
            continue  # field comparison is meaningless for the wrong type

        for name, want in case.expected.fields.items():
            value = got.value(name)
            if value is None:
                outcome.details.append(f"{name}: missing (want '{want}')")
            elif values_equal(got.document_type, name, value, want):
                correct_n += 1
            else:
                outcome.details.append(f"{name}: got '{value}', want '{want}'")
        for f in got.fields:
            if f.name not in case.expected.fields:
                outcome.details.append(f"{f.name}: unexpected value '{f.value}' (possible hallucination)")
        outcome.passed = not outcome.details

    return SuiteReport(
        "document_review",
        outcomes,
        {
            "type_accuracy": _ratio(type_ok, len(outcomes)),
            "field_precision": _ratio(correct_n, extracted_n),
            "field_recall": _ratio(correct_n, expected_n),
        },
    )


def run_checklist(make_provider: Callable[[], LLMProvider]) -> SuiteReport:
    outcomes, total, correct = [], 0, 0
    for case in _load("checklist.json", ChecklistCase):
        outcome = Outcome(case.id, passed=False)
        outcomes.append(outcome)
        total += len(case.expected)
        extractions = case.case.extraction_results()
        task = ChecklistInput(
            case_id=1,
            extractions=list(extractions.values()),
            unreviewed_document_ids=case.case.unreviewed_document_ids,
        )
        ctx = AgentContext(case.case.snapshot(), extractions)
        result = _run_agent(ChecklistAgent(make_provider()), task, ctx, outcome)
        if result is None:
            continue
        for code, want in case.expected.items():
            got = result.output.get(code)
            if got.status == want:
                correct += 1
            else:
                outcome.details.append(f"{code}: got {got.status}, want {want} ({'; '.join(got.reasons)})")
        outcome.passed = not outcome.details
    return SuiteReport("checklist", outcomes, {"requirement_accuracy": _ratio(correct, total)})


def run_next_step(make_provider: Callable[[], LLMProvider]) -> SuiteReport:
    outcomes, total, held = [], 0, 0
    for case in _load("next_step.json", NextStepCase):
        outcome = Outcome(case.id, passed=False)
        outcomes.append(outcome)
        total += len(case.must_include) + len(case.must_not_include)
        task = NextStepInput(
            case_id=1, evaluation=case.evaluation, unreviewed_document_ids=case.case.unreviewed_document_ids
        )
        ctx = AgentContext(case.case.snapshot(), case.case.extraction_results())
        result = _run_agent(NextStepAgent(make_provider()), task, ctx, outcome)
        if result is None:
            continue
        actions = result.output.actions
        for m in case.must_include:
            if any(m.matches(a) for a in actions):
                held += 1
            else:
                outcome.details.append(f"missing {m}")
        for m in case.must_not_include:
            if any(m.matches(a) for a in actions):
                outcome.details.append(f"forbidden {m} was proposed")
            else:
                held += 1
        outcome.passed = not outcome.details
    return SuiteReport("next_step", outcomes, {"property_accuracy": _ratio(held, total)})


SUITES = {"document_review": run_document_review, "checklist": run_checklist, "next_step": run_next_step}


def _ratio(n: int, d: int) -> float:
    return round(n / d, 3) if d else 1.0


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default=None, help="mock (default) or openai; falls back to LLM_PROVIDER")
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--json", type=Path, help="write the full report to this file")
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("-v", "--verbose", action="store_true", help="show details for passing cases too")
    args = parser.parse_args(argv)

    provider_name = get_provider(args.provider).name
    suites = SUITES if args.suite == "all" else {args.suite: SUITES[args.suite]}
    reports = [run(lambda: get_provider(args.provider)) for run in suites.values()]

    print(f"provider: {provider_name}\n")
    for report in reports:
        print(f"== {report.suite}  {report.passed}/{len(report.outcomes)} passed")
        for o in report.outcomes:
            mark = "PASS" if o.passed else "FAIL"
            retries = f"  retries={o.validation_retries}" if o.validation_retries else ""
            print(f"  {mark}  {o.case_id}{retries}")
            if o.error:
                print(f"        error: {o.error}")
            if not o.passed or args.verbose:
                for d in o.details:
                    print(f"        - {d}")
        print("  " + "  ".join(f"{k}={v}" for k, v in report.metrics.items()) + "\n")

    total = sum(len(r.outcomes) for r in reports)
    passed = sum(r.passed for r in reports)
    rate = _ratio(passed, total)
    print(f"overall: {passed}/{total} cases passed ({rate:.0%})")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "provider": provider_name,
                    "pass_rate": rate,
                    "suites": [
                        {"suite": r.suite, "metrics": r.metrics, "outcomes": [asdict(o) for o in r.outcomes]}
                        for r in reports
                    ],
                },
                indent=2,
            )
        )
    if rate < args.min_pass_rate:
        print(f"pass rate {rate:.0%} is below the required {args.min_pass_rate:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
