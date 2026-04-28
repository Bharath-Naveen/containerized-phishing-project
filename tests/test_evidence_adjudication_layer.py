from unittest.mock import MagicMock, patch

from src.app_v1.analyze_dashboard import (
    _apply_evidence_adjudication_layer,
    _enrich_capture_and_html_signals,
    build_dashboard_analysis,
    compute_brand_domain_coherence,
)
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


def test_brand_domain_coherence_positive_virgin_atlantic() -> None:
    out = compute_brand_domain_coherence(
        registrable_domain="virginatlantic.com",
        title="Virgin Atlantic | Book flights",
        h1="Welcome to Virgin Atlantic",
        visible_text_sample="Manage your booking and check flight status.",
    )
    assert out["brand_domain_coherence_match"] is True
    assert float(out["brand_domain_coherence_score"]) >= 0.55


def test_brand_domain_coherence_positive_coursera() -> None:
    out = compute_brand_domain_coherence(
        registrable_domain="coursera.org",
        title="Coursera | Degrees and certificates",
        h1="Learn without limits on Coursera",
        visible_text_sample="Join millions of learners worldwide.",
    )
    assert out["brand_domain_coherence_match"] is True
    assert float(out["brand_domain_coherence_score"]) >= 0.55


def test_brand_domain_coherence_negative_mismatch() -> None:
    out = compute_brand_domain_coherence(
        registrable_domain="virginatlantic.com",
        title="Coursera | Online courses",
        h1="Learn with partners",
        visible_text_sample="Google and IBM professional certificates",
    )
    assert out["brand_domain_coherence_match"] is False


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
    assert "high_risk_missing_evidence" in (out.get("evidence_phishing_signals") or [])


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


def test_official_platform_missing_evidence_high_ml_does_not_auto_escalate() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.52),
        ml={"phish_proba": 0.94, "phish_proba_calibrated": 0.94, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "framer.com",
            "final_registered_domain": "framer.com",
            "final_url": "https://www.framer.com/",
            "capture_failed": True,
        },
        html_structure_summary={},
        html_dom_summary={},
        html_structure_risk=None,
        html_dom_risk=None,
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available"},
        behavior_signals={"behavior_analysis_unavailable": True},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "uncertain"
    assert "high_risk_missing_evidence" not in (out.get("evidence_phishing_signals") or [])


def test_html_dom_unavailable_low_legitimacy_high_ml_escalates_to_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.55),
        ml={"phish_proba": 0.93, "phish_proba_calibrated": 0.93, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "weebly.com",
            "final_registered_domain": "weebly.com",
            "final_url": "https://suspicious-campaign.weebly.com/",
            "capture_failed": True,
        },
        html_structure_summary={},
        html_dom_summary={},
        html_structure_risk=None,
        html_dom_risk=None,
        html_dom_enrichment={"html_capture_missing_reason": "html_parse_partial"},
        behavior_signals={"behavior_analysis_unavailable": True},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=["suspicious_host_pattern"],
        hosting_trust={"hosting_trust_status": "hosting_trust_unknown"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert "high_risk_missing_evidence" in (out.get("evidence_phishing_signals") or [])
    assert "no_credential_capture" not in (out.get("evidence_legitimacy_signals") or [])
    assert "no_cross_domain_forms" not in (out.get("evidence_legitimacy_signals") or [])
    assert "content_rich_page" not in (out.get("evidence_legitimacy_signals") or [])


def test_coursera_like_structural_disagreement_remains_uncertain_or_legit() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.54),
        ml={"phish_proba": 0.91, "phish_proba_calibrated": 0.91, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "coursera.org",
            "final_registered_domain": "coursera.org",
            "final_url": "https://www.coursera.org/",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
        },
        html_structure_summary={"password_input_count": 0, "form_count": 1, "nav_link_count": 8, "footer_link_count": 4, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"
    assert "ml_structural_disagreement" in (out.get("evidence_ambiguity_signals") or [])


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
    assert "strong_brand_domain_mismatch" not in (out.get("evidence_phishing_signals") or [])
    assert "auth_context_on_non_official_domain" not in (out.get("evidence_phishing_signals") or [])
    assert "first_party_auth_flow_consistency" in (out.get("evidence_legitimacy_signals") or [])


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


def test_linkedin_official_authwall_oauth_mentions_do_not_trigger_brand_mismatch() -> None:
    layer2, _, _, _ = _enrich_capture_and_html_signals(
        input_url="https://www.linkedin.com/authwall",
        capture_json={
            "final_url": "https://www.linkedin.com/authwall",
            "title": "LinkedIn Login",
            "visible_text_sample": "Sign in with Google or Apple",
            "interaction": {},
        },
        soup=None,
        html_structure_summary={
            "brand_terms_found_in_text": ["google", "apple"],
            "form_count": 1,
            "password_input_count": 1,
            "email_input_count": 1,
            "phone_input_count": 0,
            "visible_text_snippet": "Continue with Google, Apple",
            "title": "LinkedIn Login",
            "h1_text": "Sign in",
            "nav_link_count": 4,
            "footer_link_count": 2,
            "has_support_help_links": True,
        },
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": True,
            "interstitial_or_preview_pattern": True,
            "page_family": "auth_login_recovery",
            "official_platform_context": "true",
        },
    )
    assert layer2.get("brand_domain_mismatch") is False
    assert layer2.get("brand_domain_mismatch_strength") in {"none", "weak"}


def test_official_content_wrapper_demoted_to_ambiguity_coursera_like() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.55),
        ml={"phish_proba": 0.93, "phish_proba_calibrated": 0.93, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "coursera.org",
            "final_registered_domain": "coursera.org",
            "final_url": "https://www.coursera.org/",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
            "password_input_external_action": False,
            "final_domain_is_free_hosting": False,
        },
        html_structure_summary={
            "password_input_count": 0,
            "email_input_count": 0,
            "form_count": 0,
            "nav_link_count": 10,
            "footer_link_count": 6,
            "has_support_help_links": True,
        },
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": True,
            "interstitial_or_preview_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.08,
        html_dom_risk=0.08,
        behavior_signals={"network_exfiltration_suspected": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert "wrapper_or_interstitial_redirect_pattern" not in (out.get("evidence_hard_blockers") or [])
    assert "official_content_wrapper_pattern" in (out.get("evidence_ambiguity_signals") or [])
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"


def test_official_content_wrapper_demoted_to_ambiguity_virgin_like() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.52),
        ml={"phish_proba": 0.88, "phish_proba_calibrated": 0.88, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "virginatlantic.com",
            "final_registered_domain": "virginatlantic.com",
            "final_url": "https://www.virginatlantic.com/",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
            "password_input_external_action": False,
            "final_domain_is_free_hosting": False,
        },
        html_structure_summary={
            "password_input_count": 0,
            "email_input_count": 0,
            "form_count": 0,
            "nav_link_count": 9,
            "footer_link_count": 7,
            "has_support_help_links": True,
        },
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": False,
            "interstitial_or_preview_pattern": True,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.09,
        html_dom_risk=0.09,
        behavior_signals={"network_exfiltration_suspected": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert "wrapper_or_interstitial_redirect_pattern" not in (out.get("evidence_hard_blockers") or [])
    assert "official_content_wrapper_pattern" in (out.get("evidence_ambiguity_signals") or [])
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"


def test_wrapper_demoted_with_brand_coherence_even_without_rich_nav() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.57),
        ml={"phish_proba": 0.91, "phish_proba_calibrated": 0.91, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "virginatlantic.com",
            "final_registered_domain": "virginatlantic.com",
            "final_url": "https://www.virginatlantic.com/en-US",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
            "brand_domain_coherence_match": True,
            "brand_domain_coherence_score": 0.8,
            "password_input_external_action": False,
            "final_domain_is_free_hosting": False,
            "security_block_page_detected": False,
        },
        html_structure_summary={"password_input_count": 0, "email_input_count": 0, "form_count": 0, "nav_link_count": 1, "footer_link_count": 0, "has_support_help_links": False},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": False,
            "interstitial_or_preview_pattern": True,
            "content_rich_profile": False,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.12,
        html_dom_risk=0.12,
        behavior_signals={"network_exfiltration_suspected": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert "wrapper_or_interstitial_redirect_pattern" not in (out.get("evidence_hard_blockers") or [])
    assert "coherent_brand_wrapper_pattern" in (out.get("evidence_ambiguity_signals") or [])
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"


def test_high_ml_clean_coherent_brand_page_caps_to_uncertain() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.56),
        ml={"phish_proba": 0.94, "phish_proba_calibrated": 0.94, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "virginatlantic.com",
            "final_registered_domain": "virginatlantic.com",
            "final_url": "https://www.virginatlantic.com/en-US",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
            "brand_domain_coherence_match": True,
            "brand_domain_coherence_score": 0.83,
            "password_input_external_action": False,
            "final_domain_is_free_hosting": False,
        },
        html_structure_summary={"password_input_count": 0, "email_input_count": 0, "form_count": 0, "nav_link_count": 9, "footer_link_count": 6, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.09,
        html_dom_risk=0.09,
        behavior_signals={"network_exfiltration_suspected": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"
    assert "brand_domain_coherence" in (out.get("evidence_legitimacy_signals") or [])
    assert "ml_brand_coherence_disagreement" in (out.get("evidence_ambiguity_signals") or [])


def test_paypal_vercel_clone_does_not_get_coherence_rescue() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.64),
        ml={"phish_proba": 0.97, "phish_proba_calibrated": 0.97, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "vercel.app",
            "final_registered_domain": "vercel.app",
            "final_url": "https://paypal-login.vercel.app/",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "brand_domain_coherence_match": False,
            "brand_domain_coherence_score": 0.0,
            "final_domain_is_free_hosting": True,
        },
        html_structure_summary={"password_input_count": 1, "email_input_count": 1, "form_count": 1},
        html_dom_summary={
            "form_action_external_domain_count": 1,
            "suspicious_credential_collection_pattern": True,
            "login_harvester_pattern": True,
            "content_rich_profile": False,
            "page_family": "auth_login_recovery",
        },
        html_structure_risk=0.8,
        html_dom_risk=0.8,
        behavior_signals={"network_exfiltration_suspected": False},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "cloud_hosted_brand_impersonation"},
        trust_blockers=["cloud_hosted_brand_impersonation", "brand_domain_mismatch"],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert "brand_domain_coherence" not in (out.get("evidence_legitimacy_signals") or [])


def test_netflix_vercel_clone_does_not_get_coherence_rescue() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.63),
        ml={"phish_proba": 0.96, "phish_proba_calibrated": 0.96, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "vercel.app",
            "final_registered_domain": "vercel.app",
            "final_url": "https://netflix-update-payment-details.vercel.app/",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "brand_domain_coherence_match": False,
            "brand_domain_coherence_score": 0.0,
            "final_domain_is_free_hosting": True,
        },
        html_structure_summary={"password_input_count": 1, "email_input_count": 1, "form_count": 1},
        html_dom_summary={
            "form_action_external_domain_count": 1,
            "suspicious_credential_collection_pattern": True,
            "login_harvester_pattern": True,
            "content_rich_profile": False,
            "page_family": "auth_login_recovery",
        },
        html_structure_risk=0.81,
        html_dom_risk=0.82,
        behavior_signals={"network_exfiltration_suspected": False},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "cloud_hosted_brand_impersonation"},
        trust_blockers=["cloud_hosted_brand_impersonation", "brand_domain_mismatch"],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_wrapper_still_hard_blocker_for_external_redirect_like_page() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.60),
        ml={"phish_proba": 0.64, "phish_proba_calibrated": 0.64, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "example.net",
            "final_registered_domain": "example.net",
            "final_url": "https://example.net/gateway",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
            "password_input_external_action": False,
        },
        html_structure_summary={"password_input_count": 0, "email_input_count": 0, "form_count": 0, "nav_link_count": 1, "footer_link_count": 0, "has_support_help_links": False},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": True,
            "interstitial_or_preview_pattern": True,
            "content_rich_profile": False,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.30,
        html_dom_risk=0.30,
        behavior_signals={"network_exfiltration_suspected": False},
        host_path_reasoning={"host_identity_class": "unknown", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "unknown"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "wrapper_or_interstitial_redirect_pattern" in (out.get("evidence_hard_blockers") or [])


def test_weebly_high_risk_missing_evidence_remains_likely_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.58),
        ml={"phish_proba": 0.95, "phish_proba_calibrated": 0.95, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "weebly.com",
            "final_registered_domain": "weebly.com",
            "final_url": "https://security-update.weebly.com/login",
            "capture_failed": True,
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "final_domain_is_free_hosting": True,
        },
        html_structure_summary={},
        html_dom_summary={},
        html_structure_risk=None,
        html_dom_risk=None,
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available"},
        behavior_signals={"behavior_analysis_unavailable": True},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=["suspicious_host_pattern", "brand_domain_mismatch"],
        hosting_trust={"hosting_trust_status": "hosting_trust_unknown"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert "high_risk_missing_evidence" in (out.get("evidence_phishing_signals") or [])


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


def test_lionic_block_page_with_strong_ml_consensus_and_brand_mismatch_is_likely_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.5),
        ml={"phish_proba": 0.62, "phish_proba_calibrated": 0.62, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "paypal-cardpaysecurity.netlify.app",
            "final_registered_domain": "lionic.com",
            "final_url": "https://block.cloud.lionic.com/blockpage/malicious.html",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "security_block_page_detected": True,
            "security_block_page_vendor": "lionic",
            "security_block_page_reasons": ["block.cloud.lionic.com", "/blockpage/malicious.html"],
        },
        html_structure_summary={"password_input_count": 0, "form_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.0,
        html_dom_risk=0.0,
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "unknown"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    ps = out.get("evidence_phishing_signals") or []
    assert "ml_consensus_strong_phishing" in ps
    assert "strong_brand_domain_mismatch" in ps
    assert "security_vendor_blocked_as_malicious" in ps
    assert "no_credential_capture" not in (out.get("evidence_legitimacy_signals") or [])
    assert "no_cross_domain_forms" not in (out.get("evidence_legitimacy_signals") or [])
    assert "content_rich_page" not in (out.get("evidence_legitimacy_signals") or [])
    assert "low_html_dom_risk" not in (out.get("evidence_legitimacy_signals") or [])


def test_lionic_block_page_with_high_ml_not_legitimate() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.5),
        ml={"phish_proba": 0.90, "phish_proba_calibrated": 0.90, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "unknown-target.example",
            "final_registered_domain": "lionic.com",
            "final_url": "https://block.cloud.lionic.com/blockpage/malicious.html",
            "brand_domain_mismatch": False,
            "security_block_page_detected": True,
            "security_block_page_vendor": "lionic",
            "security_block_page_reasons": ["this page has been blocked"],
        },
        html_structure_summary={"password_input_count": 0, "form_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.0,
        html_dom_risk=0.0,
        host_path_reasoning={},
        platform_context={"platform_context_type": "unknown"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"likely_phishing", "uncertain"}
    assert out["verdict_3way"] != "likely_legitimate"
    assert "security_block_page_observed" in (out.get("evidence_ambiguity_signals") or [])


def test_block_page_suppresses_absence_based_legitimacy_signals() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.45),
        ml={"phish_proba": 0.52, "phish_proba_calibrated": 0.52, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "leessolar.example",
            "final_registered_domain": "lionic.com",
            "final_url": "https://block.cloud.lionic.com/blockpage/malicious.html",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "security_block_page_detected": True,
            "security_block_page_vendor": "lionic",
            "security_block_page_reasons": ["might steal your confidential information"],
        },
        html_structure_summary={"password_input_count": 0, "form_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.0,
        html_dom_risk=0.0,
        host_path_reasoning={},
        platform_context={"platform_context_type": "unknown"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    ls = out.get("evidence_legitimacy_signals") or []
    assert "no_credential_capture" not in ls
    assert "no_cross_domain_forms" not in ls
    assert "content_rich_page" not in ls
    assert "low_html_dom_risk" not in ls


def test_idtech_style_incidental_brand_terms_not_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.5),
        ml={"phish_proba": 0.72, "phish_proba_calibrated": 0.72, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "idtech.com",
            "final_registered_domain": "idtech.com",
            "final_url": "https://www.idtech.com/courses",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "weak",
            "incidental_content_brand_context": True,
        },
        html_structure_summary={"password_input_count": 0, "form_count": 0, "nav_link_count": 8, "footer_link_count": 4, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.05,
        html_dom_risk=0.05,
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"likely_legitimate", "uncertain"}
    assert out["verdict_3way"] != "likely_phishing"
    assert "strong_brand_domain_mismatch" not in (out.get("evidence_phishing_signals") or [])
    assert "content_rich_page" in (out.get("evidence_legitimacy_signals") or [])


def test_brand_in_url_path_non_official_host_keeps_strong_mismatch() -> None:
    layer2, _, _, _ = _enrich_capture_and_html_signals(
        input_url="https://example.com/paypal-login",
        capture_json={
            "final_url": "https://example.com/paypal-login",
            "title": "Welcome",
            "visible_text_sample": "content",
            "interaction": {},
        },
        soup=None,
        html_structure_summary={
            "brand_terms_found_in_text": ["paypal"],
            "form_count": 0,
            "password_input_count": 0,
            "email_input_count": 0,
            "phone_input_count": 0,
            "visible_text_snippet": "content",
            "title": "Welcome",
        },
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "generic_landing"},
    )
    assert layer2.get("host_path_brand_context") is True
    assert layer2.get("brand_domain_mismatch_strength") == "strong"


def test_brand_in_title_h1_auth_context_keeps_strong_mismatch() -> None:
    layer2, _, _, _ = _enrich_capture_and_html_signals(
        input_url="https://evil.example/login",
        capture_json={
            "final_url": "https://evil.example/login",
            "title": "Microsoft Security Alert",
            "visible_text_sample": "Verify account now",
            "interaction": {},
        },
        soup=None,
        html_structure_summary={
            "brand_terms_found_in_text": ["microsoft"],
            "form_count": 1,
            "password_input_count": 1,
            "email_input_count": 1,
            "phone_input_count": 0,
            "visible_text_snippet": "Verify your Microsoft account",
            "title": "Microsoft Security Alert",
        },
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "auth_login_recovery"},
    )
    assert layer2.get("auth_payment_brand_context") is True
    assert layer2.get("brand_domain_mismatch_strength") == "strong"


def test_brand_mentions_on_same_domain_content_rich_page_are_suppressed() -> None:
    layer2, _, _, _ = _enrich_capture_and_html_signals(
        input_url="https://www.coursera.org/",
        capture_json={
            "final_url": "https://www.coursera.org/",
            "title": "Coursera | Learn from Google, IBM and Microsoft",
            "visible_text_sample": "Explore courses from world-class partners.",
            "interaction": {},
        },
        soup=None,
        html_structure_summary={
            "brand_terms_found_in_text": ["google", "ibm", "microsoft"],
            "form_count": 0,
            "password_input_count": 0,
            "email_input_count": 0,
            "phone_input_count": 0,
            "visible_text_snippet": "Catalog and partner listing only.",
            "title": "Coursera catalog",
            "h1_text": "Find your next course",
            "nav_link_count": 10,
            "footer_link_count": 8,
        },
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "page_family": "generic_landing",
            "content_rich_profile": True,
            "trust_surface_brand_domain_mismatch": False,
            "title_brand_domain_mismatch": False,
            "h1_brand_domain_mismatch": False,
            "logo_domain_mismatch": False,
            "anchor_strong_mismatch_count": 0,
            "branded_resource_domains_non_official": 4,
        },
    )
    assert layer2.get("brand_domain_mismatch") is False
    assert layer2.get("brand_domain_mismatch_strength") == "none"
    assert layer2.get("brand_mismatch_suppressed_same_domain_content_rich") is True


def test_official_page_with_third_party_references_remains_legitimate() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.35),
        ml={"phish_proba": 0.20, "phish_proba_calibrated": 0.20, "model_agreement": {"ml_consensus": "strong_legitimate"}},
        layer2_capture={
            "input_registered_domain": "google.com",
            "final_registered_domain": "google.com",
            "final_url": "https://workspace.google.com/products/gmail/",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "weak",
            "resource_only_brand_context": True,
        },
        html_structure_summary={"password_input_count": 0, "form_count": 0, "nav_link_count": 8, "footer_link_count": 4, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.08,
        html_dom_risk=0.08,
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"likely_legitimate", "uncertain"}
    assert out["verdict_3way"] != "likely_phishing"


def test_strong_brand_mismatch_alone_is_not_added_as_phishing_signal() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.45),
        ml={"phish_proba": 0.55, "phish_proba_calibrated": 0.55, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "example.com",
            "final_registered_domain": "example.com",
            "final_url": "https://example.com/",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "host_path_brand_context": False,
            "brand_in_subdomain_or_path_but_not_registered_domain": False,
            "final_domain_is_free_hosting": False,
        },
        html_structure_summary={"password_input_count": 0, "form_count": 0, "nav_link_count": 4, "footer_link_count": 2, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.18,
        html_dom_risk=0.14,
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "unknown"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert "strong_brand_domain_mismatch" not in (out.get("evidence_phishing_signals") or [])


def test_idtech_subpage_incidental_social_resource_terms_are_weak_mismatch() -> None:
    layer2, _, _, _ = _enrich_capture_and_html_signals(
        input_url="https://www.idtech.com/courses/tutoring-battlebots-at-home-with-vex-iq-kit",
        capture_json={
            "final_url": "https://www.idtech.com/courses/tutoring-battlebots-at-home-with-vex-iq-kit",
            "title": "BattleBots tutoring course",
            "visible_text_sample": "Learn robotics with references to Google, Microsoft, Facebook and WhatsApp communities.",
            "interaction": {},
        },
        soup=None,
        html_structure_summary={
            "brand_terms_found_in_text": ["google", "microsoft", "facebook", "whatsapp"],
            "form_count": 2,
            "password_input_count": 0,
            "email_input_count": 1,
            "phone_input_count": 0,
            "visible_text_snippet": "Course page with social/share references and resources.",
            "title": "BattleBots tutoring course",
            "h1_text": "Tutoring: BattleBots at Home with VEX IQ",
        },
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "page_family": "generic_landing",
            "content_rich_profile": True,
            "trust_surface_brand_domain_mismatch": False,
            "title_brand_domain_mismatch": False,
            "h1_brand_domain_mismatch": False,
            "logo_domain_mismatch": False,
            "anchor_strong_mismatch_count": 0,
            "branded_resource_domains_non_official": 2,
        },
    )
    assert layer2.get("brand_domain_mismatch_strength") in {"weak", "none"}
    assert layer2.get("auth_payment_brand_context") is False
    assert layer2.get("resource_only_brand_context") or layer2.get("incidental_content_brand_context")


def test_unifiedmentor_podia_404_not_phishing_from_ml_alone() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.55),
        ml={"phish_proba": 0.78, "phish_proba_calibrated": 0.78, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "podia.com",
            "final_registered_domain": "podia.com",
            "final_url": "https://unifiedmentor.podia.com/sessions",
            "title": "404 Not Found",
            "visible_text_sample": "This page was not found",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
        },
        html_structure_summary={"password_input_count": 0, "form_count": 0, "nav_link_count": 4, "footer_link_count": 2, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": False,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.1,
        html_dom_risk=0.1,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "platform_hosted_legitimate_candidate"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"
    assert "platform_404_or_inactive" in (out.get("evidence_ambiguity_signals") or [])


def test_platform_hosted_same_domain_newsletter_form_not_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.45),
        ml={"phish_proba": 0.52, "phish_proba_calibrated": 0.52, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "podia.com",
            "final_registered_domain": "podia.com",
            "final_url": "https://creator.podia.com/newsletter",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
        },
        html_structure_summary={"password_input_count": 0, "form_count": 1, "email_input_count": 1, "nav_link_count": 6, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.08,
        html_dom_risk=0.08,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "platform_hosted_legitimate_candidate"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"
    assert "platform_hosted_brand_consistency" in (out.get("evidence_legitimacy_signals") or [])


def test_paypal_podia_major_brand_impersonation_remains_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.6),
        ml={"phish_proba": 0.84, "phish_proba_calibrated": 0.84, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "podia.com",
            "final_registered_domain": "podia.com",
            "final_url": "https://paypal.podia.com/login",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "final_domain_is_free_hosting": False,
        },
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": False,
            "page_family": "auth_login_recovery",
        },
        html_structure_risk=0.2,
        html_dom_risk=0.2,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert "strong_brand_domain_mismatch" in (out.get("evidence_phishing_signals") or [])


def test_netflix_update_vercel_remains_likely_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.58),
        ml={"phish_proba": 0.86, "phish_proba_calibrated": 0.86, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "vercel.app",
            "final_registered_domain": "vercel.app",
            "final_url": "https://netflix-update-payment-details.vercel.app/login",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "final_domain_is_free_hosting": True,
        },
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": False,
            "page_family": "auth_login_recovery",
        },
        html_structure_risk=0.22,
        html_dom_risk=0.22,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "cloud_hosted_brand_impersonation"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_platform_404_with_major_brand_surface_not_softened() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.60),
        ml={"phish_proba": 0.88, "phish_proba_calibrated": 0.88, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "podia.com",
            "final_registered_domain": "podia.com",
            "final_url": "https://creator.podia.com/paypal-reset",
            "title": "PayPal verification 404",
            "visible_text_sample": "Page not found",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
        },
        html_structure_summary={"password_input_count": 0, "form_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": False,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.1,
        html_dom_risk=0.1,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "platform_hosted_legitimate_candidate"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_platform_404_with_suspicious_js_not_softened() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.60),
        ml={"phish_proba": 0.86, "phish_proba_calibrated": 0.86, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "podia.com",
            "final_registered_domain": "podia.com",
            "final_url": "https://creator.podia.com/missing-page",
            "title": "404 not found",
            "visible_text_sample": "The page is unavailable",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
        },
        html_structure_summary={"password_input_count": 0, "form_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": False,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.1,
        html_dom_risk=0.1,
        behavior_signals={
            "network_exfiltration_suspected": False,
            "behavior_analysis_unavailable": False,
            "js_dynamic_form_injection_detected": True,
        },
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "platform_hosted_legitimate_candidate"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_no_phishing_guard_dominates_ml_without_hard_blockers() -> None:
    out = _apply_evidence_adjudication_layer(
        {**_base_verdict(0.62), "no_phishing_evidence_guard": True},
        ml={"phish_proba": 0.91, "phish_proba_calibrated": 0.91, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "framer.com",
            "final_registered_domain": "framer.com",
            "final_url": "https://www.framer.com/?utm_source=microsoft",
            "title": "Framer - Build and ship websites",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "trust_surface_brand_context": False,
            "host_path_brand_context": False,
            "auth_payment_brand_context": False,
        },
        html_structure_summary={"password_input_count": 0, "form_count": 1, "nav_link_count": 8, "footer_link_count": 4, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
            "trust_surface_brand_domain_mismatch": False,
        },
        html_structure_risk=0.08,
        html_dom_risk=0.08,
        behavior_signals={
            "network_exfiltration_suspected": False,
            "behavior_analysis_unavailable": False,
            "network_unrelated_domains": ["a.com", "b.com", "c.com", "d.com", "e.com", "f.com"],
        },
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"
    assert "strong_brand_domain_mismatch" not in (out.get("evidence_phishing_signals") or [])
    assert "many_unrelated_third_party_domains" not in (out.get("evidence_phishing_signals") or [])


def test_official_platform_login_oauth_context_not_strong_mismatch() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.52),
        ml={"phish_proba": 0.73, "phish_proba_calibrated": 0.73, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "framer.com",
            "final_registered_domain": "framer.com",
            "final_url": "https://www.framer.com/login",
            "title": "Sign in to Framer",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "trust_surface_brand_context": False,
            "host_path_brand_context": False,
            "auth_payment_brand_context": False,
        },
        html_structure_summary={"password_input_count": 0, "form_count": 1},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": False,
            "page_family": "auth_login_recovery",
            "trust_surface_brand_domain_mismatch": False,
        },
        html_structure_risk=0.12,
        html_dom_risk=0.12,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "official_brand_auth", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_login"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert "strong_brand_domain_mismatch" not in (out.get("evidence_phishing_signals") or [])


def test_handshake_first_party_login_high_ml_caps_to_uncertain() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.60),
        ml={"phish_proba": 0.99, "phish_proba_calibrated": 0.99, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "joinhandshake.com",
            "final_registered_domain": "joinhandshake.com",
            "final_url": "https://app.joinhandshake.com/login",
            "title": "Handshake Login",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
            "final_domain_is_free_hosting": False,
            "capture_failed": False,
        },
        html_structure_summary={"password_input_count": 1, "form_count": 1, "nav_link_count": 4, "footer_link_count": 2, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": False,
            "page_family": "auth_login_recovery",
        },
        html_structure_risk=0.15,
        html_dom_risk=0.15,
        html_dom_enrichment={"html_capture_missing_reason": None},
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "unknown"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "uncertain"
    assert "first_party_login_ml_disagreement" in (out.get("evidence_ambiguity_signals") or [])
    assert "first_party_auth_flow_consistency" in (out.get("evidence_legitimacy_signals") or [])


def test_unifiedmentor_podia_404_safe_context_uncertain_not_phishing() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.57),
        ml={"phish_proba": 0.83, "phish_proba_calibrated": 0.83, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "podia.com",
            "final_registered_domain": "podia.com",
            "final_url": "https://unifiedmentor.podia.com/404",
            "title": "404 Not Found - Unified Mentor",
            "visible_text_sample": "Visit unifiedmentor.com and subscribe to newsletter",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "trust_surface_brand_context": False,
            "host_path_brand_context": False,
            "auth_payment_brand_context": False,
        },
        html_structure_summary={"password_input_count": 0, "form_count": 1, "email_input_count": 1, "nav_link_count": 5, "footer_link_count": 3, "has_support_help_links": True},
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
        behavior_signals={
            "network_exfiltration_suspected": False,
            "behavior_analysis_unavailable": False,
            "network_unrelated_domains": ["cdn.podia.com", "stripe.com", "paypal.com", "cloudflare.com", "jsdelivr.net", "google-analytics.com"],
        },
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "creator_platform_404_or_inactive"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "uncertain"
    assert "strong_brand_domain_mismatch" not in (out.get("evidence_phishing_signals") or [])
    assert "many_unrelated_third_party_domains" not in (out.get("evidence_phishing_signals") or [])


def test_unifiedmentor_podia_root_creator_uncertainty_cap_applies() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.59),
        ml={"phish_proba": 0.88, "phish_proba_calibrated": 0.88, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "podia.com",
            "final_registered_domain": "podia.com",
            "final_url": "https://unifiedmentor.podia.com/",
            "title": "Unified Mentor | Courses and Mentorship",
            "visible_text_sample": "Join our newsletter. Visit unifiedmentor.com for more details.",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "weak",
            "trust_surface_brand_context": False,
            "host_path_brand_context": False,
            "auth_payment_brand_context": False,
        },
        html_structure_summary={
            "password_input_count": 0,
            "form_count": 1,
            "email_input_count": 1,
            "nav_link_count": 7,
            "footer_link_count": 3,
            "has_support_help_links": True,
            "h1_text": "Unified Mentor",
        },
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
            "trust_surface_brand_domain_mismatch": False,
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        behavior_signals={
            "network_exfiltration_suspected": False,
            "behavior_analysis_unavailable": False,
            "network_unrelated_domains": ["cdn.podia.com", "stripe.com", "paypal.com", "cloudflare.com", "google-analytics.com"],
        },
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "platform_hosted_legitimate_candidate"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "uncertain"
    assert "strong_brand_domain_mismatch" not in (out.get("evidence_phishing_signals") or [])


def test_linkedin_official_login_gets_official_domain_trust_prior() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.56),
        ml={"phish_proba": 0.78, "phish_proba_calibrated": 0.78, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "linkedin.com",
            "final_registered_domain": "linkedin.com",
            "final_url": "https://www.linkedin.com/login",
            "brand_domain_mismatch": False,
            "final_domain_is_free_hosting": False,
        },
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "auth_login_recovery"},
        html_structure_risk=0.2,
        html_dom_risk=0.18,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_login"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"
    assert "official_domain_trust_prior" in (out.get("evidence_legitimacy_signals") or [])


def test_coursera_official_page_gets_official_domain_trust_prior() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.56),
        ml={"phish_proba": 0.75, "phish_proba_calibrated": 0.75, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={"input_registered_domain": "coursera.org", "final_registered_domain": "coursera.org", "final_url": "https://www.coursera.org/courseraplus/", "brand_domain_mismatch": False, "final_domain_is_free_hosting": False},
        html_structure_summary={"password_input_count": 0, "form_count": 1, "nav_link_count": 6, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "content_rich_profile": True, "page_family": "generic_landing"},
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert "official_domain_trust_prior" in (out.get("evidence_legitimacy_signals") or [])


def test_official_netflix_gets_official_domain_trust_prior() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.52),
        ml={"phish_proba": 0.68, "phish_proba_calibrated": 0.68, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={"input_registered_domain": "netflix.com", "final_registered_domain": "netflix.com", "final_url": "https://www.netflix.com/", "brand_domain_mismatch": False, "final_domain_is_free_hosting": False},
        html_structure_summary={"password_input_count": 0, "form_count": 0, "nav_link_count": 6, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "content_rich_profile": True, "page_family": "generic_landing"},
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert "official_domain_trust_prior" in (out.get("evidence_legitimacy_signals") or [])


def test_virgin_atlantic_gets_official_domain_trust_prior_and_not_phishing_when_clean() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.58),
        ml={"phish_proba": 0.91, "phish_proba_calibrated": 0.91, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "virginatlantic.com",
            "final_registered_domain": "virginatlantic.com",
            "final_url": "https://www.virginatlantic.com/en-US",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
            "brand_domain_coherence_match": True,
            "brand_domain_coherence_score": 0.8,
            "password_input_external_action": False,
            "final_domain_is_free_hosting": False,
        },
        html_structure_summary={"password_input_count": 0, "email_input_count": 0, "form_count": 0, "nav_link_count": 8, "footer_link_count": 5, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": False,
            "interstitial_or_preview_pattern": True,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert "official_domain_trust_prior" in (out.get("evidence_legitimacy_signals") or [])
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}
    assert out["verdict_3way"] != "likely_phishing"


def test_virgin_atlantic_strong_ml_overconfidence_relaxes_to_likely_legitimate() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.60),
        ml={"phish_proba": 0.95, "phish_proba_calibrated": 0.95, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "virginatlantic.com",
            "final_registered_domain": "virginatlantic.com",
            "final_url": "https://www.virginatlantic.com/en-US",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
            "brand_domain_coherence_match": True,
            "brand_domain_coherence_score": 0.86,
            "password_input_external_action": False,
            "final_domain_is_free_hosting": False,
        },
        html_structure_summary={"password_input_count": 0, "email_input_count": 0, "form_count": 0, "nav_link_count": 9, "footer_link_count": 6, "has_support_help_links": True},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "content_rich_profile": True,
            "page_family": "generic_landing",
        },
        html_structure_risk=0.08,
        html_dom_risk=0.08,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_legitimate"
    assert "ml_overconfidence_relaxed_due_to_strong_legitimacy" in (out.get("evidence_ambiguity_signals") or [])


def test_vercel_clone_does_not_get_official_domain_trust_prior() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.6),
        ml={"phish_proba": 0.95, "phish_proba_calibrated": 0.95, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={"input_registered_domain": "vercel.app", "final_registered_domain": "vercel.app", "final_url": "https://paypal-login.vercel.app/", "brand_domain_mismatch": True, "brand_domain_mismatch_strength": "strong", "final_domain_is_free_hosting": True},
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "auth_login_recovery"},
        html_structure_risk=0.2,
        html_dom_risk=0.2,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "cloud_hosted_brand_impersonation"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "official_domain_trust_prior" not in (out.get("evidence_legitimacy_signals") or [])
    assert out["verdict_3way"] == "likely_phishing"


def test_github_io_amazon_clone_does_not_get_official_domain_trust_prior() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.58),
        ml={"phish_proba": 0.90, "phish_proba_calibrated": 0.90, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={"input_registered_domain": "github.io", "final_registered_domain": "github.io", "final_url": "https://amazon-login.github.io/", "brand_domain_mismatch": True, "brand_domain_mismatch_strength": "strong", "final_domain_is_free_hosting": True},
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "auth_login_recovery"},
        html_structure_risk=0.2,
        html_dom_risk=0.2,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "cloud_hosted_brand_impersonation"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "official_domain_trust_prior" not in (out.get("evidence_legitimacy_signals") or [])
    assert out["verdict_3way"] == "likely_phishing"


def test_myshopify_store_does_not_get_official_domain_trust_prior() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.5),
        ml={"phish_proba": 0.6, "phish_proba_calibrated": 0.6, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={"input_registered_domain": "myshopify.com", "final_registered_domain": "myshopify.com", "final_url": "https://sample-store.myshopify.com/", "brand_domain_mismatch": False, "final_domain_is_free_hosting": False},
        html_structure_summary={"password_input_count": 0, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "generic_landing"},
        html_structure_risk=0.12,
        html_dom_risk=0.12,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "unknown", "host_legitimacy_confidence": "medium"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_unknown"},
        legitimacy_bundle={},
    )
    assert "official_domain_trust_prior" not in (out.get("evidence_legitimacy_signals") or [])


def test_https_official_page_gets_valid_https_transport_but_not_sole_legit() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.58),
        ml={"phish_proba": 0.92, "phish_proba_calibrated": 0.92, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "coursera.org",
            "final_registered_domain": "coursera.org",
            "final_url": "https://www.coursera.org/courseraplus/",
            "brand_domain_mismatch": False,
            "final_domain_is_free_hosting": False,
            "uses_https": True,
            "tls_or_cert_error_detected": False,
            "insecure_scheme_detected": False,
        },
        html_structure_summary={"password_input_count": 0, "form_count": 1, "nav_link_count": 6, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "content_rich_profile": True, "page_family": "generic_landing"},
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert "valid_https_transport" in (out.get("evidence_legitimacy_signals") or [])
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}


def test_https_phishing_clone_remains_likely_phishing_with_valid_https_transport() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.60),
        ml={"phish_proba": 0.95, "phish_proba_calibrated": 0.95, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "vercel.app",
            "final_registered_domain": "vercel.app",
            "final_url": "https://netflix-update-payment-details.vercel.app/",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
            "final_domain_is_free_hosting": True,
            "uses_https": True,
            "tls_or_cert_error_detected": False,
            "insecure_scheme_detected": False,
        },
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "auth_login_recovery"},
        html_structure_risk=0.20,
        html_dom_risk=0.20,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern", "host_legitimacy_confidence": "low"},
        platform_context={"platform_context_type": "cloud_hosted_brand_impersonation"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert "valid_https_transport" in (out.get("evidence_legitimacy_signals") or [])


def test_http_login_gets_insecure_auth_surface() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.55),
        ml={"phish_proba": 0.62, "phish_proba_calibrated": 0.62, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "example.com",
            "final_registered_domain": "example.com",
            "final_url": "http://example.com/login",
            "brand_domain_mismatch": False,
            "uses_https": False,
            "insecure_scheme_detected": True,
            "tls_or_cert_error_detected": False,
        },
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "auth_login_recovery"},
        html_structure_risk=0.25,
        html_dom_risk=0.22,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "unknown", "host_legitimacy_confidence": "medium"},
        platform_context={"platform_context_type": "unknown"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "insecure_auth_surface" in (out.get("evidence_phishing_signals") or [])


def test_cert_error_login_gets_insecure_auth_surface() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.55),
        ml={"phish_proba": 0.66, "phish_proba_calibrated": 0.66, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={
            "input_registered_domain": "example.com",
            "final_registered_domain": "example.com",
            "final_url": "https://example.com/login",
            "brand_domain_mismatch": False,
            "uses_https": True,
            "insecure_scheme_detected": False,
            "tls_or_cert_error_detected": True,
        },
        html_structure_summary={"password_input_count": 1, "form_count": 1},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "auth_login_recovery"},
        html_structure_risk=0.25,
        html_dom_risk=0.22,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "unknown", "host_legitimacy_confidence": "medium"},
        platform_context={"platform_context_type": "unknown"},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert "insecure_auth_surface" in (out.get("evidence_phishing_signals") or [])


def test_valid_https_does_not_suppress_hard_blocker() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.55),
        ml={"phish_proba": 0.40, "phish_proba_calibrated": 0.40, "model_agreement": {"ml_consensus": "split"}},
        layer2_capture={"input_registered_domain": "x.com", "final_registered_domain": "x.com", "final_url": "https://x.com/login", "uses_https": True, "tls_or_cert_error_detected": False, "insecure_scheme_detected": False},
        html_structure_summary={"password_input_count": 1},
        html_dom_summary={"form_action_external_domain_count": 1, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "page_family": "auth_login_recovery"},
        html_structure_risk=0.5,
        html_dom_risk=0.4,
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={},
        platform_context={},
        trust_blockers=[],
        hosting_trust={},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert "valid_https_transport" in (out.get("evidence_legitimacy_signals") or [])


def test_official_domain_ml_overconfidence_caps_to_uncertain() -> None:
    out = _apply_evidence_adjudication_layer(
        _base_verdict(0.60),
        ml={"phish_proba": 0.98, "phish_proba_calibrated": 0.98, "model_agreement": {"ml_consensus": "strong_phishing"}},
        layer2_capture={
            "input_registered_domain": "coursera.org",
            "final_registered_domain": "coursera.org",
            "final_url": "https://www.coursera.org/",
            "brand_domain_mismatch": False,
            "final_domain_is_free_hosting": False,
            "uses_https": True,
            "tls_or_cert_error_detected": False,
            "insecure_scheme_detected": False,
        },
        html_structure_summary={"password_input_count": 0, "form_count": 1, "nav_link_count": 6, "footer_link_count": 3, "has_support_help_links": True},
        html_dom_summary={"form_action_external_domain_count": 0, "suspicious_credential_collection_pattern": False, "login_harvester_pattern": False, "content_rich_profile": True, "page_family": "generic_landing"},
        html_structure_risk=0.10,
        html_dom_risk=0.10,
        html_dom_enrichment={"html_capture_missing_reason": None},
        behavior_signals={"network_exfiltration_suspected": False, "behavior_analysis_unavailable": False},
        host_path_reasoning={"host_identity_class": "known_platform_official", "host_legitimacy_confidence": "high"},
        platform_context={"platform_context_type": "official_platform_domain"},
        trust_blockers=[],
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
        legitimacy_bundle={},
    )
    assert out["verdict_3way"] == "uncertain"
    assert "official_domain_ml_overconfidence_suspected" in (out.get("evidence_ambiguity_signals") or [])
