"""AI-adjustment tests removed from tests/test_ml_capture_miss_safety.py when the AI layer was archived."""

from src.app_v1.ai_adjudicator import apply_ai_adjustment, should_run_ai_adjudication
from src.app_v1.verdict_policy import Verdict3WayConfig


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