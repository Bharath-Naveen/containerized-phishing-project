"""Global invariant: ML phishing + capture/HTML gap + no trust anchor → never likely_legitimate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from phishguard.app.dashboard import _apply_ml_phishing_capture_miss_legitimacy_safety, build_dashboard_analysis
from phishguard.app.schemas import CaptureResult
from phishguard.app.verdict_policy import Verdict3WayConfig


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

    with patch("phishguard.app.dashboard.PipelineConfig.from_env", return_value=cfg):
        with patch("phishguard.app.dashboard.capture_url", return_value=fake):
            with patch("phishguard.app.dashboard.predict_layer1", return_value=fake_ml):
                with patch("phishguard.app.dashboard.no_phishing_evidence_guard", return_value=True):
                    out, _gaps = build_dashboard_analysis(
                        "https://evil-phish-login-verify.example/fake",
                        reinforcement=True,
                    )
    verdict = out.get("verdict") or {}
    assert verdict.get("verdict_3way") == "uncertain"
    assert verdict.get("ml_phishing_capture_miss_safety_applied") is True
    assert verdict.get("combined_score_pre_ml_capture_miss_safety") is not None
    assert any("Deterministic safety guardrail applied" in r for r in (verdict.get("reasons") or []))
