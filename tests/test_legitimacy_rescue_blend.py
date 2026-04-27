from src.app_v1.analyze_dashboard import (
    _apply_dns_feature_dominance_dampening,
    _apply_legitimacy_rescue_on_verdict,
)
from src.app_v1.config import PipelineConfig
from src.app_v1.verdict_policy import Verdict3WayConfig


def _base_cfg() -> PipelineConfig:
    return PipelineConfig()


def _base_verdict() -> dict:
    return {"combined_score": 0.76, "reasons": [], "label": "likely_phishing", "verdict_3way": "likely_phishing"}


def _base_ml() -> dict:
    return {"phish_proba": 0.82, "predicted_phishing": True}


def _base_capture() -> dict:
    return {
        "input_registered_domain": "example.com",
        "final_registered_domain": "example.com",
        "redirect_chain_registered_domains": ["example.com", "example.com"],
        "cross_domain_redirect_count": 0,
        "brand_domain_mismatch": False,
        "final_domain_is_free_hosting": False,
        "contains_punycode": False,
        "contains_non_ascii": False,
        "final_url": "https://app.example.com/login",
    }


def _base_html_structure() -> dict:
    return {"password_input_count": 1}


def _base_dom() -> dict:
    return {
        "trust_action_context": False,
        "suspicious_credential_collection_pattern": False,
        "form_action_external_domain_count": 0,
        "login_harvester_pattern": False,
        "wrapper_page_pattern": False,
        "interstitial_or_preview_pattern": False,
        "trust_surface_brand_domain_mismatch": False,
        "anchor_strong_mismatch_count": 0,
    }


def _base_hp() -> dict:
    return {
        "host_identity_class": "enterprise_app",
        "host_legitimacy_confidence": "high",
        "path_fit_assessment": "plausible",
    }


def _base_bundle() -> dict:
    return {
        "suspicious_form_action_cross_origin": False,
        "no_deceptive_token_placement": True,
        "no_free_hosting_signal": True,
    }


def test_legitimate_app_subdomain_login_flow_downgrades_to_uncertain() -> None:
    hp = {
        "host_identity_class": "enterprise_app",
        "host_legitimacy_confidence": "high",
        "path_fit_assessment": "plausible",
    }
    out = _apply_legitimacy_rescue_on_verdict(
        _base_verdict(),
        ml=_base_ml(),
        host_path_reasoning=hp,
        html_structure_summary=_base_html_structure(),
        html_structure_risk=0.10,
        html_dom_summary=_base_dom(),
        html_dom_risk=0.12,
        layer2_capture=_base_capture(),
        legitimacy_bundle=_base_bundle(),
        cfg=_base_cfg(),
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["legitimacy_rescue_applied"] is True
    assert out["verdict_3way"] == "uncertain"
    assert "redirect_chain_domain_consistent" in (out.get("legitimacy_rescue_reasons") or [])


def test_same_domain_login_with_cross_domain_form_action_not_rescued() -> None:
    dom = dict(_base_dom())
    dom["form_action_external_domain_count"] = 1
    out = _apply_legitimacy_rescue_on_verdict(
        _base_verdict(),
        ml=_base_ml(),
        host_path_reasoning=_base_hp(),
        html_structure_summary=_base_html_structure(),
        html_structure_risk=0.10,
        html_dom_summary=dom,
        html_dom_risk=0.12,
        layer2_capture=_base_capture(),
        legitimacy_bundle=_base_bundle(),
        cfg=_base_cfg(),
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["legitimacy_rescue_applied"] is False
    assert "cross_domain_form_action" in (out.get("legitimacy_rescue_blockers") or [])


def test_free_hosted_brand_impersonation_not_rescued() -> None:
    cap = dict(_base_capture())
    cap["final_domain_is_free_hosting"] = True
    bundle = dict(_base_bundle())
    bundle["no_free_hosting_signal"] = False
    out = _apply_legitimacy_rescue_on_verdict(
        _base_verdict(),
        ml=_base_ml(),
        host_path_reasoning=_base_hp(),
        html_structure_summary=_base_html_structure(),
        html_structure_risk=0.08,
        html_dom_summary=_base_dom(),
        html_dom_risk=0.10,
        layer2_capture=cap,
        legitimacy_bundle=bundle,
        cfg=_base_cfg(),
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["legitimacy_rescue_applied"] is False
    assert "free_hosting_impersonation" in (out.get("legitimacy_rescue_blockers") or [])


def test_punycode_or_non_ascii_host_not_rescued() -> None:
    cap = dict(_base_capture())
    cap["contains_punycode"] = True
    out = _apply_legitimacy_rescue_on_verdict(
        _base_verdict(),
        ml=_base_ml(),
        host_path_reasoning=_base_hp(),
        html_structure_summary=_base_html_structure(),
        html_structure_risk=0.08,
        html_dom_summary=_base_dom(),
        html_dom_risk=0.10,
        layer2_capture=cap,
        legitimacy_bundle=_base_bundle(),
        cfg=_base_cfg(),
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["legitimacy_rescue_applied"] is False
    assert "punycode_or_non_ascii_host" in (out.get("legitimacy_rescue_blockers") or [])


def test_brand_domain_mismatch_not_rescued() -> None:
    cap = dict(_base_capture())
    cap["brand_domain_mismatch"] = True
    dom = dict(_base_dom())
    dom["trust_surface_brand_domain_mismatch"] = True
    out = _apply_legitimacy_rescue_on_verdict(
        _base_verdict(),
        ml=_base_ml(),
        host_path_reasoning=_base_hp(),
        html_structure_summary=_base_html_structure(),
        html_structure_risk=0.08,
        html_dom_summary=dom,
        html_dom_risk=0.10,
        layer2_capture=cap,
        legitimacy_bundle=_base_bundle(),
        cfg=_base_cfg(),
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["legitimacy_rescue_applied"] is False
    assert "brand_domain_mismatch" in (out.get("legitimacy_rescue_blockers") or [])


def test_rescue_blocked_for_phishing_context_wrapper_pattern() -> None:
    dom = {
        "trust_action_context": True,
        "form_action_external_domain_count": 1,
    }
    out = _apply_legitimacy_rescue_on_verdict(
        _base_verdict(),
        ml=_base_ml(),
        host_path_reasoning=_base_hp(),
        html_structure_summary=_base_html_structure(),
        html_structure_risk=0.10,
        html_dom_summary=dom,
        html_dom_risk=0.44,
        layer2_capture=_base_capture(),
        legitimacy_bundle={**_base_bundle(), "suspicious_form_action_cross_origin": True},
        cfg=_base_cfg(),
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["legitimacy_rescue_applied"] is False
    assert out["combined_score"] == 0.76


def test_handshake_style_strong_legitimacy_override_forces_uncertain() -> None:
    verdict = {"combined_score": 0.88, "reasons": [], "label": "likely_phishing", "verdict_3way": "likely_phishing"}
    ml = {"phish_proba": 0.896, "predicted_phishing": True}
    cap = {
        "input_registered_domain": "joinhandshake.com",
        "final_registered_domain": "joinhandshake.com",
        "redirect_chain_registered_domains": ["joinhandshake.com", "joinhandshake.com"],
        "cross_domain_redirect_count": 0,
        "brand_domain_mismatch": False,
        "final_domain_is_free_hosting": False,
        "contains_punycode": False,
        "contains_non_ascii": False,
        "final_url": "https://app.joinhandshake.com/login",
    }
    dom = dict(_base_dom())
    out = _apply_legitimacy_rescue_on_verdict(
        verdict,
        ml=ml,
        host_path_reasoning=_base_hp(),
        html_structure_summary=_base_html_structure(),
        html_structure_risk=0.0,
        html_dom_summary=dom,
        html_dom_risk=0.2,
        layer2_capture=cap,
        legitimacy_bundle=_base_bundle(),
        cfg=_base_cfg(),
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["legitimacy_strong_override_triggered"] is True
    assert out["legitimacy_rescue_applied"] is True
    assert out["verdict_3way"] == "uncertain"
    assert out["label"] == "uncertain"


def test_dns_dominance_dampening_applies_when_legitimacy_is_strong() -> None:
    ml = {
        "top_linear_signals": [
            {"feature": "num__dns_resolves", "signed_coef": 1.0, "value": 0.9},
            {"feature": "num__dns_error", "signed_coef": 1.1, "value": 0.7},
            {"feature": "num__url_length", "signed_coef": 0.2, "value": 0.4},
        ]
    }
    score, meta = _apply_dns_feature_dominance_dampening(
        ml_effective_score=0.9,
        ml=ml,
        layer2_capture=_base_capture(),
        html_structure_risk=0.05,
        html_dom_risk=0.10,
        legitimacy_bundle=_base_bundle(),
        cfg=_base_cfg(),
    )
    assert isinstance(score, float)
    assert score < 0.9
    assert meta.get("dns_dominant") is True
    assert meta.get("dns_dampening_applied") is True
