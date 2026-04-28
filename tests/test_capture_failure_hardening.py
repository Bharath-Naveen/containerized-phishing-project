"""Tests for capture-failure suspicion fields, verdict guardrail, and click-probe diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.app_v1.analyze_dashboard import (
    _apply_capture_failure_verdict_hardening,
    _finalize_click_probe_diagnostics_on_capture,
    _merge_capture_failure_fields,
    build_dashboard_analysis,
    no_phishing_evidence_guard,
)
from src.app_v1.schemas import CaptureResult


def test_merge_capture_failure_high_ml_moderate_suspicion() -> None:
    cap: dict = {"error": "net::ERR_NETWORK_ACCESS_DENIED", "capture_strategy": "failed", "html_path": ""}
    ml = {"phish_proba_calibrated": 0.85, "phish_proba_model_raw": 0.85, "phish_proba": 0.85}
    _merge_capture_failure_fields(cap, ml)
    assert cap["capture_failed"] is True
    assert cap["capture_failure_type"] == "network_access_denied"
    assert cap["capture_failure_suspicious"] is True
    assert cap["capture_failure_suspicion_level"] == "moderate"


def test_merge_capture_failure_low_ml_weak_not_auto_phish() -> None:
    cap = {"error": "timeout after 30s", "capture_strategy": "playwright_headless", "html_path": ""}
    ml = {"phish_proba_calibrated": 0.2, "phish_proba_model_raw": 0.2, "phish_proba": 0.2}
    _merge_capture_failure_fields(cap, ml)
    assert cap["capture_failed"] is True
    assert cap["capture_failure_type"] == "timeout"
    assert cap["capture_failure_suspicion_level"] == "weak"
    assert cap["capture_failure_suspicious"] is True


def test_verdict_hardening_blocked_by_official_trust() -> None:
    verdict = {
        "label": "uncertain",
        "verdict_3way": "uncertain",
        "confidence": "low",
        "combined_score": 0.45,
        "reasons": [],
    }
    ml = {
        "phish_proba_calibrated": 0.9,
        "phish_proba_model_raw": 0.9,
        "phish_proba": 0.9,
        "predicted_phishing": True,
    }
    bundle = {"official_registrable_anchor": True}
    out = _apply_capture_failure_verdict_hardening(
        verdict,
        ml=ml,
        legitimacy_bundle=bundle,
        capture_failed=True,
        html_missing_for_reinforcement=True,
    )
    assert out["label"] == "uncertain"
    assert out.get("capture_failure_verdict_guardrail_applied") is False


def test_verdict_hardening_upgrades_uncertain_when_eligible() -> None:
    verdict = {
        "label": "uncertain",
        "verdict_3way": "uncertain",
        "confidence": "low",
        "combined_score": 0.45,
        "reasons": [],
    }
    ml = {
        "phish_proba_calibrated": 0.75,
        "phish_proba_model_raw": 0.75,
        "phish_proba": 0.75,
        "predicted_phishing": True,
    }
    out = _apply_capture_failure_verdict_hardening(
        verdict,
        ml=ml,
        legitimacy_bundle={},
        capture_failed=True,
        html_missing_for_reinforcement=True,
    )
    assert out["capture_failure_verdict_guardrail_applied"] is True
    assert out["label"] == "likely_phishing"
    assert float(out["combined_score"]) >= 0.56


def test_click_probe_disabled_skip_reason() -> None:
    cap: dict = {"capture": {"click_probe": {}}}
    inner = cap["capture"]
    _finalize_click_probe_diagnostics_on_capture(inner, enable_click_probe=False)
    assert inner["click_probe"]["click_probe_skip_reason"] == "disabled_by_config"
    assert inner["click_probe"]["click_probe_enabled"] is False


def test_no_phishing_guard_respects_capture_failure_high_ml() -> None:
    assert (
        no_phishing_evidence_guard(
            html_structure_summary={"password_input_count": 0},
            html_dom_summary={
                "form_action_external_domain_count": 0,
                "suspicious_credential_collection_pattern": False,
                "trust_action_context": False,
                "strong_impersonation_context": False,
                "wrapper_page_pattern": False,
                "login_harvester_pattern": False,
            },
            html_dom_risk=0.12,
            host_path_reasoning={"host_legitimacy_confidence": "high", "path_fit_assessment": "plausible"},
            capture_failed=True,
            html_structure_error="missing_html_path",
            html_capture_missing_reason="html_not_available",
            ml_calibrated_phish=0.85,
        )
        is False
    )


def test_build_dashboard_capture_failure_evidence_gap() -> None:
    fake = CaptureResult(
        original_url="https://evil-phish-login-verify.example/fake",
        final_url="https://evil-phish-login-verify.example/fake",
        title="",
        screenshot_path="",
        fullpage_screenshot_path="",
        html_path="",
        visible_text="",
        error="net::ERR_NETWORK_ACCESS_DENIED",
        capture_strategy="failed",
    )

    fake_ml = {
        "phish_proba": 0.85,
        "phish_proba_model_raw": 0.85,
        "phish_proba_calibrated": 0.85,
        "predicted_phishing": True,
        "canonical_url": "https://evil-phish-login-verify.example/fake",
        "brand_structure_features": {},
        "error": None,
    }

    cfg = MagicMock()
    cfg.enable_click_probe = False

    with patch("src.app_v1.analyze_dashboard.PipelineConfig.from_env", return_value=cfg):
        with patch("src.app_v1.analyze_dashboard.capture_url", return_value=fake):
            with patch("src.app_v1.analyze_dashboard.predict_layer1", return_value=fake_ml):
                out, gaps = build_dashboard_analysis(
                    "https://evil-phish-login-verify.example/fake",
                    reinforcement=True,
                )
    cap = (out.get("reinforcement") or {}).get("capture") or {}
    assert cap.get("capture_failure_suspicious") is True
    assert "HTML/DOM analysis was unavailable because live capture did not complete." in gaps
    assert cap.get("click_probe", {}).get("click_probe_skip_reason") == "disabled_by_config"


def test_finalize_click_probe_capture_failed_before_probe() -> None:
    cap_inner: dict = {"capture_failed": True, "click_probe": {}}
    _finalize_click_probe_diagnostics_on_capture(cap_inner, enable_click_probe=True)
    assert cap_inner["click_probe"]["click_probe_skip_reason"] == "capture_failed_before_probe"


def test_build_dashboard_click_probe_disabled_config() -> None:
    """Default env keeps click probe off — diagnostics must surface disabled_by_config when capture succeeds."""
    fake = CaptureResult(
        original_url="https://example.com/",
        final_url="https://example.com/",
        title="",
        screenshot_path="",
        fullpage_screenshot_path="",
        html_path="",
        visible_text="",
        error=None,
        capture_strategy="playwright_headless",
    )

    fake_ml = {
        "phish_proba": 0.1,
        "phish_proba_model_raw": 0.1,
        "phish_proba_calibrated": 0.1,
        "predicted_phishing": False,
        "canonical_url": "https://example.com/",
        "brand_structure_features": {},
        "error": None,
    }

    cfg = MagicMock()
    cfg.enable_click_probe = False

    with patch("src.app_v1.analyze_dashboard.PipelineConfig.from_env", return_value=cfg):
        with patch("src.app_v1.analyze_dashboard.capture_url", return_value=fake):
            with patch("src.app_v1.analyze_dashboard.predict_layer1", return_value=fake_ml):
                out, _gaps = build_dashboard_analysis(
                    "https://example.com/",
                    reinforcement=True,
                )
    cp = ((out.get("reinforcement") or {}).get("capture") or {}).get("click_probe") or {}
    assert cp.get("click_probe_skip_reason") == "disabled_by_config"
