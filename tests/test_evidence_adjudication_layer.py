from unittest.mock import MagicMock, patch

from src.app_v1.analyze_dashboard import _apply_evidence_adjudication_layer, build_dashboard_analysis
from src.app_v1.schemas import CaptureResult


def _base_verdict(score: float = 0.6) -> dict:
    return {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": score, "reasons": []}


def test_high_ml_alone_does_not_convict_without_corroboration() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.62),
        ml={"phish_proba": 0.96, "phish_proba_calibrated": 0.96},
        layer2_capture={"input_registered_domain": "example.com", "final_registered_domain": "example.com"},
        html_structure_summary={"password_input_count": 0, "nav_link_count": 5, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}


def test_hard_blocker_always_convicts() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.4),
        ml={"phish_proba": 0.4, "phish_proba_calibrated": 0.4},
        layer2_capture={"input_registered_domain": "x.com", "final_registered_domain": "x.com"},
        html_structure_summary={"password_input_count": 1},
        html_dom_summary={"form_action_external_domain_count": 1, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False},
        html_structure_risk=0.5,
        html_dom_risk=0.4,
        host_path_reasoning={},
        platform_context={},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_official_content_page_can_be_legit_or_uncertain() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.52),
        ml={"phish_proba": 0.90, "phish_proba_calibrated": 0.90},
        layer2_capture={
            "input_registered_domain": "framer.com",
            "final_registered_domain": "framer.com",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "weak",
        },
        html_structure_summary={"password_input_count": 0, "nav_link_count": 6, "footer_link_count": 4, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
            "trust_surface_brand_domain_mismatch": False,
        },
        html_structure_risk=0.12,
        html_dom_risk=0.12,
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}


def test_cloud_impersonation_remains_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.58),
        ml={"phish_proba": 0.88, "phish_proba_calibrated": 0.88},
        layer2_capture={"input_registered_domain": "vercel.app", "final_registered_domain": "vercel.app"},
        html_structure_summary={"password_input_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": True, "login_harvester_pattern": True},
        html_structure_risk=0.8,
        html_dom_risk=0.8,
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern"},
        platform_context={"platform_context_type": "cloud_hosted_brand_impersonation"},
        trust_blockers=["cloud_hosted_brand_impersonation"],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_inactive_benign_page_remains_uncertain() -> None:
    out = _apply_evidence_adjudication_layer(
        {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": 0.5, "reasons": [], "inactive_site_detected": True},
        ml={"phish_proba": 0.35, "phish_proba_calibrated": 0.35},
        layer2_capture={"input_registered_domain": "example.org", "final_registered_domain": "example.org"},
        html_structure_summary={"password_input_count": 0},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False},
        html_structure_risk=0.05,
        html_dom_risk=0.05,
        host_path_reasoning={"host_identity_class": "unknown"},
        platform_context={"platform_context_type": "unknown"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_unknown"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "uncertain"


def test_dormant_suspicious_hosted_page_remains_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        {
            "label": "likely_phishing",
            "verdict_3way": "likely_phishing",
            "combined_score": 0.66,
            "reasons": [],
            "dormant_phishing_infra_detected": True,
        },
        ml={"phish_proba": 0.62, "phish_proba_calibrated": 0.62},
        layer2_capture={"input_registered_domain": "weebly.com", "final_registered_domain": "weebly.com"},
        html_structure_summary={"password_input_count": 0},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False},
        html_structure_risk=0.18,
        html_dom_risk=0.18,
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=["suspicious_host_pattern", "dormant_phishing_infra"],
        hosting_trust={"hosting_trust_status": "hosting_trust_unknown"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_netflix_style_brand_subdomain_evasive_case_is_likely_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        {"label": "uncertain", "verdict_3way": "uncertain", "combined_score": 0.55, "reasons": []},
        ml={"phish_proba": 0.992, "phish_proba_calibrated": 0.992},
        layer2_capture={
            "input_registered_domain": "2ndstage.app",
            "final_registered_domain": "2ndstage.app",
            "final_url": "https://netflix.2ndstage.app/vote/plesh/mF9QhKN5lIUREAnavsgC",
            "capture_failed": True,
            "capture_failure_suspicious": True,
        },
        html_structure_summary={},
        html_dom_summary={},
        html_structure_risk=None,
        html_dom_risk=None,
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available"},
        host_path_reasoning={
            "host_identity_class": "suspicious_host_pattern",
            "host_legitimacy_confidence": "low",
        },
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=["suspicious_host_pattern"],
        hosting_trust={"hosting_trust_status": "hosting_trust_unknown"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert "high_ml_score" in (out.get("evidence_phishing_signals") or [])
    assert "suspicious_host_pattern" in (out.get("evidence_phishing_signals") or [])
    assert "brand_subdomain_impersonation" in (out.get("evidence_phishing_signals") or [])
    assert "capture_failure_suspicious" in (out.get("evidence_phishing_signals") or [])
    assert "no_credential_capture" not in (out.get("evidence_legitimacy_signals") or [])
    assert "no_cross_domain_forms" not in (out.get("evidence_legitimacy_signals") or [])
    assert "html_dom_unavailable" in (out.get("evidence_ambiguity_signals") or [])


def test_capture_failed_high_ml_suspicious_host_is_likely_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.56),
        ml={"phish_proba": 0.97, "phish_proba_calibrated": 0.97},
        layer2_capture={
            "input_registered_domain": "example.app",
            "final_registered_domain": "example.app",
            "final_url": "https://random.example.app/a",
            "capture_failed": True,
        },
        html_structure_summary={},
        html_dom_summary={},
        html_structure_risk=None,
        html_dom_risk=None,
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available"},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=["suspicious_host_pattern"],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_capture_failed_benign_trusted_domain_remains_uncertain() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.50),
        ml={"phish_proba": 0.60, "phish_proba_calibrated": 0.60},
        layer2_capture={
            "input_registered_domain": "example.com",
            "final_registered_domain": "example.com",
            "final_url": "https://www.example.com/",
            "capture_failed": True,
        },
        html_structure_summary={},
        html_dom_summary={},
        html_structure_risk=None,
        html_dom_risk=None,
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available"},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "uncertain"
    assert "no_credential_capture" not in (out.get("evidence_legitimacy_signals") or [])
    assert "no_cross_domain_forms" not in (out.get("evidence_legitimacy_signals") or [])


def test_brand_subdomain_impersonation_strong_signal_not_hard_blocker_by_itself() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.50),
        ml={"phish_proba": 0.65, "phish_proba_calibrated": 0.65},
        layer2_capture={
            "input_registered_domain": "2ndstage.app",
            "final_registered_domain": "2ndstage.app",
            "final_url": "https://netflix.2ndstage.app/landing",
            "capture_failed": False,
        },
        html_structure_summary={"password_input_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        html_dom_enrichment={"html_capture_missing_reason": None},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=["suspicious_host_pattern"],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "brand_subdomain_impersonation" in (out.get("evidence_phishing_signals") or [])
    assert "brand_subdomain_impersonation" not in (out.get("evidence_hard_blockers") or [])


def test_same_domain_consistency_not_counted_for_suspicious_host() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.50),
        ml={"phish_proba": 0.70, "phish_proba_calibrated": 0.70},
        layer2_capture={
            "input_registered_domain": "2ndstage.app",
            "final_registered_domain": "2ndstage.app",
            "final_url": "https://netflix.2ndstage.app/",
        },
        html_structure_summary={"password_input_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        html_dom_enrichment={"html_capture_missing_reason": None},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=["suspicious_host_pattern"],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "same_domain_consistency" not in (out.get("evidence_legitimacy_signals") or [])


def test_no_openai_key_required_when_ai_disabled() -> None:
    fake = CaptureResult(
        original_url="https://www.example.com",
        final_url="https://www.example.com",
        title="Example",
        screenshot_path="",
        fullpage_screenshot_path="",
        html_path="",
        visible_text="Example content",
        error="",
        capture_strategy="http_fallback",
    )
    fake_ml = {
        "phish_proba": 0.4,
        "phish_proba_model_raw": 0.4,
        "phish_proba_calibrated": 0.4,
        "predicted_phishing": False,
        "canonical_url": "https://www.example.com",
        "brand_structure_features": {},
        "error": None,
    }
    cfg = MagicMock()
    cfg.enable_click_probe = False
    with patch("src.app_v1.analyze_dashboard.PipelineConfig.from_env", return_value=cfg):
        with patch("src.app_v1.analyze_dashboard.capture_url", return_value=fake):
            with patch("src.app_v1.analyze_dashboard.predict_layer1", return_value=fake_ml):
                out, _ = build_dashboard_analysis("https://www.example.com", reinforcement=True)
    assert (out.get("verdict") or {}).get("evidence_adjudication_applied") is True
    assert "ai_adjudication" not in (out.get("verdict") or {})


def test_frontend_has_no_ai_toggle() -> None:
    from pathlib import Path

    p = Path("src/app_v1/frontend.py")
    txt = p.read_text(encoding="utf-8")
    assert "Enable optional AI analyst notes" not in txt
    assert "AI adjudication" not in txt
    assert "optional AI" not in txt
    assert "AI analyst" not in txt
    assert "OpenAI" not in txt
    assert "LLM" not in txt


def test_dashboard_cli_has_no_ai_flag() -> None:
    from pathlib import Path

    p = Path("src/app_v1/analyze_dashboard.py")
    txt = p.read_text(encoding="utf-8")
    assert "--ai" not in txt
    assert "OPENAI_API_KEY" not in txt


def test_official_authwall_wrapper_not_hard_blocker_linkedin_case() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.30),
        ml={
            "phish_proba": 0.229,
            "phish_proba_calibrated": 0.229,
            "model_agreement": {
                "ml_consensus": "strong_legitimate",
                "ml_prob_spread": 0.06,
                "ml_model_votes_phishing": 0,
                "ml_model_votes_legitimate": 4,
            },
        },
        layer2_capture={
            "input_registered_domain": "linkedin.com",
            "final_registered_domain": "linkedin.com",
            "final_url": "https://www.linkedin.com/in/bharath-naveen/",
            "brand_domain_mismatch": False,
            "final_domain_is_free_hosting": False,
        },
        html_structure_summary={"password_input_count": 1},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": True,
            "interstitial_or_preview_pattern": True,
            "content_rich_profile": False,
            "page_family": "auth_login_recovery",
        },
        html_structure_risk=0.0,
        html_dom_risk=0.0,
        host_path_reasoning={"host_identity_class": "official_brand_auth", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"likely_legitimate", "uncertain"}
    assert out["verdict_3way"] != "likely_phishing"
    assert "wrapper_or_interstitial_redirect_pattern" not in (out.get("evidence_hard_blockers") or [])


def test_official_authwall_wrapper_is_ambiguity_not_hard_blocker() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.40),
        ml={"phish_proba": 0.60, "phish_proba_calibrated": 0.60, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "linkedin.com",
            "final_registered_domain": "linkedin.com",
            "final_url": "https://www.linkedin.com/authwall",
            "brand_domain_mismatch": False,
        },
        html_structure_summary={"password_input_count": 1},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": True,
            "interstitial_or_preview_pattern": False,
        },
        html_structure_risk=0.0,
        html_dom_risk=0.0,
        host_path_reasoning={"host_identity_class": "official_brand_auth", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_login"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "wrapper_or_interstitial_redirect_pattern" not in (out.get("evidence_hard_blockers") or [])
    assert "official_authwall_wrapper_pattern" in (out.get("evidence_ambiguity_signals") or [])


def test_free_hosted_amazon_clone_with_strong_ml_consensus_escalates_to_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.55),
        ml={
            "phish_proba": 0.654,
            "phish_proba_calibrated": 0.654,
            "model_agreement": {"ml_consensus": "strong_phishing", "ml_prob_spread": 0.10},
        },
        layer2_capture={
            "input_registered_domain": "github.io",
            "final_registered_domain": "github.io",
            "final_url": "https://muhammadbilal0011.github.io/CodeAlpha_Task3/",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "final_domain_is_free_hosting": True,
        },
        html_structure_summary={"password_input_count": 0, "nav_link_count": 8, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.05,
        html_dom_risk=0.05,
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=["brand_domain_mismatch"],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    ps = out.get("evidence_phishing_signals") or []
    assert "ml_consensus_strong_phishing" in ps
    assert "strong_brand_domain_mismatch" in ps
    assert "free_hosted_brand_impersonation" in ps


def test_content_rich_does_not_rescue_free_hosted_brand_impersonation() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.55),
        ml={"phish_proba": 0.66, "phish_proba_calibrated": 0.66, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "github.io",
            "final_registered_domain": "github.io",
            "final_url": "https://evilbrand.github.io/",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "final_domain_is_free_hosting": True,
        },
        html_structure_summary={"password_input_count": 0, "nav_link_count": 10, "footer_link_count": 4, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.01,
        html_dom_risk=0.01,
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=["brand_domain_mismatch"],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "content_rich_page" not in (out.get("evidence_legitimacy_signals") or [])
    assert "low_html_dom_risk" not in (out.get("evidence_legitimacy_signals") or [])
    assert out["verdict_3way"] == "likely_phishing"


def test_official_google_and_amazon_pages_remain_likely_legitimate() -> None:
    for final_url in ("https://accounts.google.com/signin", "https://www.amazon.com/ap/signin"):
        out = _apply_evidence_adjudication_layer(
            _base_verdict(0.30),
            ml={"phish_proba": 0.18, "phish_proba_calibrated": 0.18, "model_agreement": {"ml_consensus": "strong_legitimate"}},
            layer2_capture={
                "input_registered_domain": final_url.split("/")[2].split(".", 1)[-1],
                "final_registered_domain": final_url.split("/")[2].split(".", 1)[-1],
                "final_url": final_url,
                "brand_domain_mismatch": False,
                "final_domain_is_free_hosting": False,
            },
            html_structure_summary={"password_input_count": 1, "nav_link_count": 6, "footer_link_count": 3, "has_support_help_links": True},
            html_dom_summary={
                "form_action_external_domain_count": 0,
                "suspicious_credential_collection_pattern": False,
                "login_harvester_pattern": False,
                "content_rich_profile": False,
                "page_family": "auth_login_recovery",
            },
            html_structure_risk=0.05,
            html_dom_risk=0.05,
            host_path_reasoning={"host_identity_class": "official_brand_auth", "host_legitimacy_confidence": "high"},
            platform_context={"platform_context_type": "official_platform_login"},
            trust_blockers=[],
            hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
            legitimacy_bundle={},
        )
        assert out["verdict_3way"] in {"likely_legitimate", "uncertain"}
        assert out["verdict_3way"] != "likely_phishing"
