"""Smoke test for AI adjudication compact audit outputs."""

import pytest

from src.pipeline.paths import models_dir, reports_dir


def test_ai_adjudication_audit_writes_reports() -> None:
    if not (models_dir() / "layer1_primary.joblib").is_file():
        pytest.skip("No layer1_primary model present")
    from src.pipeline.ai_adjudication_audit import run_ai_adjudication_audit

    rep = run_ai_adjudication_audit(reinforcement=False, layer1_use_dns=False)
    assert "summary" in rep
    assert "per_suite" in rep
    assert rep["summary"]["n"] >= 1
    for suite in ("obvious_legit", "tricky_legit", "obvious_phish", "hard_phishing"):
        assert suite in rep["per_suite"]
    assert (reports_dir() / "ai_adjudication_audit.json").is_file()
    assert (reports_dir() / "ai_adjudication_audit.md").is_file()
