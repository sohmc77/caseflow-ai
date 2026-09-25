"""Deterministic requirement rules.

Checks like date arithmetic, amount thresholds, and name matching belong in
code, not in a prompt. Models are unreliable at them, and code can be
unit-tested.
"""

import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from .schemas import (
    CaseSnapshot,
    DocumentExtractionResult,
    MinAmountRule,
    MinValidityRule,
    NameMatchesApplicantRule,
    RequiredFieldsRule,
    RequirementSnapshot,
    Rule,
    RuleCheckResult,
)


def check_requirement(
    requirement: RequirementSnapshot, extraction: DocumentExtractionResult, snapshot: CaseSnapshot
) -> RuleCheckResult:
    failures: list[str] = []
    if extraction.document_type not in requirement.accepted_document_types:
        failures.append(
            f"document type '{extraction.document_type}' is not accepted "
            f"(expected one of {[t.value for t in requirement.accepted_document_types]})"
        )
    else:
        for rule in requirement.rules:
            failures.extend(_check(rule, extraction, snapshot))
    return RuleCheckResult(
        requirement_code=requirement.code,
        document_id=extraction.document_id,
        passed=not failures,
        failures=failures,
    )


def describe(rule: Rule) -> str:
    match rule:
        case RequiredFieldsRule(fields=fields):
            return f"fields present: {', '.join(fields)}"
        case MinValidityRule(field=field, days=days):
            return f"{field} at least {days} days after the case date"
        case MinAmountRule(field=field, amount=amount, currency=currency):
            return f"{field} at least {amount} {currency}"
        case NameMatchesApplicantRule(field=field):
            return f"{field} matches the applicant's name"
    raise TypeError(f"unknown rule {rule!r}")


def _check(rule: Rule, ex: DocumentExtractionResult, snapshot: CaseSnapshot) -> list[str]:
    match rule:
        case RequiredFieldsRule(fields=fields):
            return [f"missing field '{f}'" for f in fields if ex.value(f) is None]

        case MinValidityRule(field=field, days=days):
            raw = ex.value(field)
            if raw is None:
                return [f"missing field '{field}'"]
            deadline = snapshot.as_of + timedelta(days=days)
            if date.fromisoformat(raw) < deadline:
                return [f"{field} {raw} is before the required {deadline.isoformat()}"]
            return []

        case MinAmountRule(field=field, amount=amount, currency=currency, currency_field=cur_field):
            raw, cur = ex.value(field), ex.value(cur_field)
            if raw is None:
                return [f"missing field '{field}'"]
            if cur != currency:
                return [f"{cur_field} is '{cur}', expected '{currency}' (no currency conversion)"]
            try:
                if Decimal(raw) < amount:
                    return [f"{field} {raw} {cur} is below the required {amount} {currency}"]
            except InvalidOperation:
                return [f"{field} '{raw}' is not a number"]
            return []

        case NameMatchesApplicantRule(field=field):
            raw = ex.value(field)
            if raw is None:
                return [f"missing field '{field}'"]
            if _name_tokens(raw) != _name_tokens(snapshot.applicant_name):
                return [f"{field} '{raw}' does not match applicant '{snapshot.applicant_name}'"]
            return []
    raise TypeError(f"unknown rule {rule!r}")


def _name_tokens(name: str) -> list[str]:
    """Order-insensitive: 'OKAFOR, AMARA' matches 'Amara Okafor'."""
    return sorted(re.findall(r"\w+", name.casefold()))
