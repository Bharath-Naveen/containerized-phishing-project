"""Global invariant: ML phishing + capture/HTML gap + no trust anchor → never likely_legitimate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.app_v1.ai_adjudicator import (
    apply_ai_adjustment,
    should_run_ai_adjudication,
)
from src.app_v1.analyze_dashboard import _apply_ml_phishing_capture_miss_legitimacy_safety, build_dashboard_analysis
from src.app_v1.schemas import CaptureResult
from src.app_v1.verdict_policy import Verdict3WayConfig


def _base_verdict_legit(combined: float) -> dict:
    return {
        "label": "likely_legitimate",
        "verdict_3way": "likely_legitimate",
        "confidence": "medium",
        "combined_score": combined,
        "reasons": ["prior"],
    }


def test_safety_never_allows_likely_legitimate_when_ml_phish_capture_miss() -> None:
    ml = {
        "predicted_phishing": True,
        "phish_proba_calibrated": 0.55,
        "phish_proba_model_raw": 0.55,
        "phish_proba": 0.55,
    }
    out = _apply_ml_phishing_capture_miss_legitimacy_safety(
        _base_verdict_legit(0.32),
        ml=ml,
        legitimacy_bundle={},
        capture_failed=True,
        html_capture_missing_reason="html_not_available",
        html_structure_error=None,
        html_dom_anomaly_error=None,
    )
    assert out["verdict_3way"] != "likely_legitimate"
    assert out["label"] == "uncertain"
    assert out.get("ml_phishing_capture_miss_safety_applied") is True
    assert any("Deterministic safety guardrail applied" in r for r in (out.get("reasons") or []))
    assert out.get("combined_score_pre_ml_capture_miss_safety") == 0.32


def test_safety_uncertain_band_for_calibrated_50_to_70() -> None:
    ml = {
        "predicted_phishing": True,
        "phish_proba_calibrated": 0.62,
        "phish_proba_model_raw": 0.62,
        "phish_proba": 0.62,
    }
    out = _apply_ml_phishing_capture_miss_legitimacy_safety(
        _base_verdict_legit(0.30),
        ml=ml,
        legitimacy_bundle={},
        capture_failed=True,
        html_capture_missing_reason=None,
        html_structure_error="missing_html_path",
        html_dom_anomaly_error=None,
    )
    assert out["label"] == "uncertain"
    lo, hi = Verdict3WayConfig().combined_low, Verdict3WayConfig().combined_high
    assert lo < float(out["combined_score"]) < hi


def test_safety_allows_likely_phishing_when_calibrated_ge_70() -> None:
    ml = {
        "predicted_phishing": True,
        "phish_proba_calibrated": 0.78,
        "phish_proba_model_raw": 0.78,
        "phish_proba": 0.78,
    }
    out = _apply_ml_phishing_capture_miss_legitimacy_safety(
        _base_verdict_legit(0.30),
        ml=ml,
        legitimacy_bundle={},
        capture_failed=True,
        html_capture_missing_reason="html_not_available",
        html_structure_error=None,
        html_dom_anomaly_error="missing_html_path",
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert float(out["combined_score"]) >= Verdict3WayConfig().combined_high


def test_safety_skipped_for_official_trust_anchor() -> None:
    ml = {
        "predicted_phishing": True,
        "phish_proba_calibrated": 0.55,
        "phish_proba_model_raw": 0.55,
        "phish_proba": 0.55,
    }
    v0 = _base_verdict_legit(0.32)
    out = _apply_ml_phishing_capture_miss_legitimacy_safety(
        v0,
        ml=ml,
        legitimacy_bundle={"official_registrable_anchor": True},
        capture_failed=True,
        html_capture_missing_reason="html_not_available",
        html_structure_error=None,
        html_dom_anomaly_error=None,
    )
    assert out["label"] == v0["label"] and out["combined_score"] == v0["combined_score"]
    assert not out.get("ml_phishing_capture_miss_safety_applied")


def test_safety_at_least_uncertain_when_cal_below_50() -> None:
    ml = {
        "predicted_phishing": True,
        "phish_proba_calibrated": 0.42,
        "phish_proba_model_raw": 0.42,
        "phish_proba": 0.42,
    }
    out = _apply_ml_phishing_capture_miss_legitimacy_safety(
        _base_verdict_legit(0.28),
        ml=ml,
        legitimacy_bundle={},
        capture_failed=True,
        html_capture_missing_reason="html_not_available",
        html_structure_error=None,
        html_dom_anomaly_error=None,
    )
    assert out["label"] == "uncertain"
    assert out["confidence"] == "low"


def test_should_run_ai_forces_ml_capture_miss_review() -> None:
    should, reasons = should_run_ai_adjudication(
        pre_ai_combined=0.32,
        ml_effective_score=0.8,
        org_risk_adjusted=0.05,
        bundle={},
        pre_verdict="likely_legitimate",
        input_url="https://evil.example/login",
        force_ml_phishing_capture_miss_review=True,
    )
    assert should is True
    assert "ml_predicted_phishing_but_pre_verdict_legitimate" in reasons


def test_apply_ai_blocks_downward_adjustment_under_capture_miss_review() -> None:
    cfg = Verdict3WayConfig(combined_low=0.38, combined_high=0.56)
    ctx = {
        "ml_phishing_capture_miss_review": {"block_ai_legitimizing_adjustment": True},
    }
    out = apply_ai_adjustment(
        pre_ai_score=0.35,
        pre_ai_verdict="likely_legitimate",
        ai_result={"adjustment_direction": "down", "adjustment_magnitude": 0.12},
        adjudication_context=ctx,
        verdict_cfg=cfg,
    )
    assert out["post_ai_verdict"] == "uncertain"
    assert float(out["post_ai_score"]) >= 0.35
    assert float(out["post_ai_score"]) > float(out["pre_ai_score"])


def test_apply_ai_clamps_post_score_if_still_likely_legitimate() -> None:
    cfg = Verdict3WayConfig(combined_low=0.38, combined_high=0.56)
    ctx = {"ml_phishing_capture_miss_review": {"block_ai_legitimizing_adjustment": True}}
    out = apply_ai_adjustment(
        pre_ai_score=0.36,
        pre_ai_verdict="likely_legitimate",
        ai_result={"adjustment_direction": "none", "adjustment_magnitude": 0.0},
        adjudication_context=ctx,
        verdict_cfg=cfg,
    )
    assert out["post_ai_verdict"] == "uncertain"
    assert cfg.combined_low < float(out["post_ai_score"]) < cfg.combined_high


def test_deterministic_safety_with_ai_disabled_after_no_phishing_override() -> None:
    """Safety must run before final output even when AI adjudication is off."""
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
        "phish_proba": 0.55,
        "phish_proba_model_raw": 0.55,
        "phish_proba_calibrated": 0.55,
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
                with patch("src.app_v1.analyze_dashboard.no_phishing_evidence_guard", return_value=True):
                    out, _gaps = build_dashboard_analysis(
                        "https://evil-phish-login-verify.example/fake",
                        reinforcement=True,
                        ai_adjudication=False,
                    )
    verdict = out.get("verdict") or {}
    assert verdict.get("verdict_3way") == "uncertain"
    assert verdict.get("ml_phishing_capture_miss_safety_applied") is True
    assert verdict.get("combined_score_pre_ml_capture_miss_safety") is not None
    assert any("Deterministic safety guardrail applied" in r for r in (verdict.get("reasons") or []))
    ai = verdict.get("ai_adjudication") or {}
    assert ai.get("ran") is not True
