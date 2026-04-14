"""Defensive tests for bounded AI adjudication behavior."""

from src.app_v1.ai_adjudicator import apply_ai_adjustment, should_run_ai_adjudication
from src.app_v1.verdict_policy import Verdict3WayConfig


def test_ai_runs_for_uncertain_band() -> None:
    should, reasons = should_run_ai_adjudication(
        pre_ai_combined=0.48,
        ml_effective_score=0.62,
        org_risk_adjusted=0.20,
        bundle={},
        pre_verdict="uncertain",
        input_url="https://example.com/path",
    )
    assert should is True
    assert reasons


def test_bounded_ai_cannot_flip_strong_high_into_legitimate() -> None:
    cfg = Verdict3WayConfig(combined_low=0.38, combined_high=0.56)
    out = apply_ai_adjustment(
        pre_ai_score=0.92,
        pre_ai_verdict="likely_phishing",
        ai_result={"adjustment_direction": "down", "adjustment_magnitude": 0.15},
        verdict_cfg=cfg,
    )
    assert out["post_ai_score"] >= cfg.combined_high
    assert out["post_ai_verdict"] == "likely_phishing"


def test_narrow_borderline_legit_down_boost_can_flip() -> None:
    cfg = Verdict3WayConfig(combined_low=0.38, combined_high=0.56)
    out = apply_ai_adjustment(
        pre_ai_score=0.4975,
        pre_ai_verdict="uncertain",
        ai_result={
            "adjustment_direction": "down",
            "adjustment_magnitude": 0.10,
            "ai_assessment": "likely_legitimate",
            "ai_confidence": "medium",
        },
        adjudication_context={
            "layer1_ml": {"phish_proba_raw": 0.76},
            "reinforcement": {"redirect_count": 0, "cross_domain_redirect_count": 0},
            "legitimacy_bundle": {
                "no_free_hosting_signal": True,
                "no_brand_mismatch": True,
                "suspicious_form_action_cross_origin": False,
            },
        },
        verdict_cfg=cfg,
    )
    assert out["narrow_legit_down_boost_applied"] is True
    assert out["post_ai_score"] < 0.38
    assert out["post_ai_verdict"] == "likely_legitimate"


def test_ai_runs_for_high_legit_content_over_suspicious_ml() -> None:
    should, reasons = should_run_ai_adjudication(
        pre_ai_combined=0.61,
        ml_effective_score=0.66,
        org_risk_adjusted=0.08,
        bundle={"no_brand_mismatch": True, "no_free_hosting_signal": True, "no_deceptive_token_placement": True},
        pre_verdict="likely_phishing",
        input_url="https://example.com/article/test",
        html_dom_anomaly_summary={
            "page_family": "article_news",
            "trust_action_context": False,
            "suspicious_credential_collection_pattern": False,
            "form_action_external_domain_count": 0,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": False,
            "interstitial_or_preview_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
            "anchor_strong_mismatch_count": 0,
        },
        host_path_reasoning={
            "host_legitimacy_confidence": "high",
            "path_fit_assessment": "plausible",
            "host_identity_class": "public_content_platform",
        },
    )
    assert should is True
    assert "high_legit_content_over_suspicious_ml" in reasons


def test_high_legit_content_rescue_boost_applies_when_safe() -> None:
    cfg = Verdict3WayConfig(combined_low=0.38, combined_high=0.56)
    out = apply_ai_adjustment(
        pre_ai_score=0.62,
        pre_ai_verdict="likely_phishing",
        ai_result={
            "adjustment_direction": "down",
            "adjustment_magnitude": 0.12,
            "ai_assessment": "likely_legitimate",
            "ai_confidence": "high",
        },
        adjudication_context={
            "host_path_reasoning": {
                "host_legitimacy_confidence": "high",
                "path_fit_assessment": "plausible",
                "host_identity_class": "public_content_platform",
            },
            "html_dom_anomaly": {
                "summary": {
                    "page_family": "article_news",
                    "trust_action_context": False,
                    "suspicious_credential_collection_pattern": False,
                    "form_action_external_domain_count": 0,
                    "login_harvester_pattern": False,
                    "wrapper_page_pattern": False,
                    "interstitial_or_preview_pattern": False,
                    "trust_surface_brand_domain_mismatch": False,
                    "anchor_strong_mismatch_count": 0,
                }
            },
            "legitimacy_bundle": {
                "suspicious_form_action_cross_origin": False,
                "no_deceptive_token_placement": True,
                "no_free_hosting_signal": True,
            },
        },
        verdict_cfg=cfg,
    )
    assert out["high_legit_content_down_boost_applied"] is True
    assert out["adjustment_applied"] < 0.0
