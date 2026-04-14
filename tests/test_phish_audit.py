"""Phishing full-system audit structure (no Playwright by default in test)."""

import pytest

from src.pipeline.paths import models_dir, reports_dir


def test_run_phish_audit_writes_json_and_has_regression_keys() -> None:
    if not (models_dir() / "layer1_primary.joblib").is_file():
        pytest.skip("No layer1_primary.joblib")

    from src.pipeline.phish_audit import OUTPUT_JSON, run_phish_audit

    rep = run_phish_audit(reinforcement=False, layer1_use_dns=False)
    assert "summary" in rep
    assert "per_suite" in rep
    assert "regression_checks" in rep
    assert rep["summary"]["n"] >= 1
    for k in ("obvious_phish", "hard_phishing"):
        assert k in rep["per_suite"]
    assert "strong_layer1_now_uncertain" in rep["regression_checks"]
    outp = reports_dir() / OUTPUT_JSON
    assert outp.is_file()
    assert (reports_dir() / "phish_audit_full_system.md").is_file()
