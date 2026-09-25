from agents import rules

from .conftest import make_extraction


def requirement(snapshot, code):
    return snapshot.requirement(code)


def test_passport_valid_and_name_matches_in_any_order(snapshot):
    ex = make_extraction(
        10, "passport", full_name="OKAFOR, AMARA", passport_number="AR4471902", expiry_date="2031-03-14"
    )
    result = rules.check_requirement(requirement(snapshot, "identity"), ex, snapshot)
    assert result.passed, result.failures


def test_validity_window_boundary(snapshot):
    # as_of 2026-09-01 + 180 days = 2027-02-28
    ok = make_extraction(10, "passport", full_name="Amara Okafor", passport_number="X", expiry_date="2027-02-28")
    late = make_extraction(10, "passport", full_name="Amara Okafor", passport_number="X", expiry_date="2027-02-27")
    req = requirement(snapshot, "identity")
    assert rules.check_requirement(req, ok, snapshot).passed
    failures = rules.check_requirement(req, late, snapshot).failures
    assert failures == ["expiry_date 2027-02-27 is before the required 2027-02-28"]


def test_funds_below_threshold_and_wrong_currency(snapshot):
    req = requirement(snapshot, "proof_of_funds")
    low = make_extraction(
        11, "bank_statement", account_holder="Amara Okafor", closing_balance="8250.00", currency="EUR"
    )
    usd = make_extraction(11, "bank_statement", account_holder="Amara Okafor", closing_balance="99999", currency="USD")
    assert "below the required 10000 EUR" in rules.check_requirement(req, low, snapshot).failures[0]
    assert "no currency conversion" in rules.check_requirement(req, usd, snapshot).failures[0]


def test_name_mismatch(snapshot):
    ex = make_extraction(11, "bank_statement", account_holder="Lucas Brennan", closing_balance="20000", currency="EUR")
    failures = rules.check_requirement(requirement(snapshot, "proof_of_funds"), ex, snapshot).failures
    assert failures == ["account_holder 'Lucas Brennan' does not match applicant 'Amara Okafor'"]


def test_wrong_document_type_short_circuits(snapshot):
    ex = make_extraction(11, "bank_statement", account_holder="Amara Okafor")
    result = rules.check_requirement(requirement(snapshot, "identity"), ex, snapshot)
    assert not result.passed
    assert len(result.failures) == 1 and "not accepted" in result.failures[0]
