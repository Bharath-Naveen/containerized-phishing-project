"""Three-way verdict thresholds."""

from src.app_v1.verdict_policy import Verdict3WayConfig, verdict_3way


def test_verdict_three_bands() -> None:
    cfg = Verdict3WayConfig(combined_low=0.38, combined_high=0.56)
    assert verdict_3way(0.30, cfg)[0] == "likely_legitimate"
    assert verdict_3way(0.45, cfg)[0] == "uncertain"
    assert verdict_3way(0.60, cfg)[0] == "likely_phishing"
