"""Locks in the mock baseline so a regression in the harness or the mock shows up in CI."""

from evals import run
from evals.run import SUITES


def test_mock_baseline():
    reports = {name: suite(lambda: run.get_provider("mock")) for name, suite in SUITES.items()}

    failing = {o.case_id for r in reports.values() for o in r.outcomes if not o.passed}
    # The heuristic mock cannot read prose; a real model is expected to pass these.
    assert failing == {"employment_letter_prose", "degree_certificate_ceremonial"}
    assert reports["document_review"].metrics["field_precision"] == 1.0  # misses, but never invents
    assert reports["checklist"].metrics["requirement_accuracy"] == 1.0


def test_cli_gate_fails_below_threshold(tmp_path, capsys):
    report = tmp_path / "report.json"
    assert run.main(["--provider", "mock", "--json", str(report), "--min-pass-rate", "0.8"]) == 0
    assert run.main(["--provider", "mock", "--min-pass-rate", "0.99"]) == 1
    assert '"pass_rate"' in report.read_text()
