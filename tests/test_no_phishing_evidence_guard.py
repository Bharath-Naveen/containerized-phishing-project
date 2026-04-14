from src.app_v1.analyze_dashboard import (
    _apply_no_phishing_evidence_override,
    no_phishing_evidence_guard,
)


def test_no_phishing_evidence_guard_true() -> None:
    hs = {"password_input_count": 0}
    dom = {
        "form_action_external_domain_count": 0,
        "suspicious_credential_collection_pattern": False,
        "trust_action_context": False,
        "strong_impersonation_context": False,
        "wrapper_page_pattern": False,
        "login_harvester_pattern": False,
    }
    hp = {"host_legitimacy_confidence": "high", "path_fit_assessment": "plausible"}
    assert no_phishing_evidence_guard(
        html_structure_summary=hs,
        html_dom_summary=dom,
        html_dom_risk=0.12,
        host_path_reasoning=hp,
    ) is True


def test_no_phishing_evidence_override_forces_legitimate() -> None:
    out = _apply_no_phishing_evidence_override(
        {"combined_score": 0.62, "reasons": []},
        guard_triggered=True,
    )
    assert out["label"] == "likely_legitimate"
    assert out["verdict_3way"] == "likely_legitimate"
    assert out["combined_score"] <= 0.35
    assert out["legitimacy_rescue_applied"] is True


def test_no_phishing_evidence_guard_false_on_any_red_flag() -> None:
    hs = {"password_input_count": 1}
    dom = {
        "form_action_external_domain_count": 0,
        "suspicious_credential_collection_pattern": False,
        "trust_action_context": False,
        "strong_impersonation_context": False,
        "wrapper_page_pattern": False,
        "login_harvester_pattern": False,
    }
    hp = {"host_legitimacy_confidence": "high", "path_fit_assessment": "plausible"}
    assert no_phishing_evidence_guard(
        html_structure_summary=hs,
        html_dom_summary=dom,
        html_dom_risk=0.12,
        host_path_reasoning=hp,
    ) is False
