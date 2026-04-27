from src.app_v1.analyze_dashboard import (
    _apply_inactive_site_overlay,
    _apply_platform_context_policy,
    _apply_untrusted_builder_hosting_downgrade,
    _classify_platform_host_context,
    _apply_hosting_trust_promotion,
    _apply_legitimacy_rescue_on_verdict,
    _detect_cloud_hosted_brand_impersonation,
    _evaluate_hosting_domain_trust,
)
from src.app_v1.config import PipelineConfig
from src.app_v1.verdict_policy import Verdict3WayConfig


def _cfg() -> PipelineConfig:
    return PipelineConfig()


def _verdict_uncertain() -> dict:
    return {"combined_score": 0.5, "label": "uncertain", "verdict_3way": "uncertain", "reasons": []}


def _verdict_phish() -> dict:
    return {"combined_score": 0.82, "label": "likely_phishing", "verdict_3way": "likely_phishing", "reasons": []}


def _safe_dom() -> dict:
    return {
        "form_action_external_domain_count": 0,
        "trust_surface_brand_domain_mismatch": False,
        "suspicious_credential_collection_pattern": False,
        "login_harvester_pattern": False,
        "wrapper_page_pattern": False,
        "interstitial_or_preview_pattern": False,
        "anchor_strong_mismatch_count": 0,
    }


def _safe_bundle() -> dict:
    return {
        "suspicious_form_action_cross_origin": False,
        "no_deceptive_token_placement": True,
        "no_free_hosting_signal": True,
    }


def test_official_domain_handshake_can_be_promoted_with_safe_signals() -> None:
    trust = _evaluate_hosting_domain_trust(
        layer2_capture={
            "input_registered_domain": "joinhandshake.com",
            "final_registered_domain": "joinhandshake.com",
            "final_url": "https://app.joinhandshake.com/login",
        },
        html_structure_risk=0.1,
        html_dom_risk=0.1,
        blockers=[],
        cfg=_cfg(),
    )
    out = _apply_hosting_trust_promotion(
        _verdict_uncertain(),
        trust=trust,
        ml={"phish_proba": 0.6},
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["verdict_3way"] in {"uncertain", "likely_legitimate"}


def test_handshake_login_vercel_app_cloud_impersonation_detected() -> None:
    assert _detect_cloud_hosted_brand_impersonation(
        final_registered_domain="vercel.app",
        final_host="handshake-login.vercel.app",
        final_url="https://handshake-login.vercel.app/login",
    )
    out = _apply_legitimacy_rescue_on_verdict(
        {"combined_score": 0.8, "label": "likely_phishing", "verdict_3way": "likely_phishing", "reasons": []},
        ml={"phish_proba": 0.9, "predicted_phishing": True},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern"},
        html_structure_summary={"password_input_count": 1},
        html_structure_risk=0.0,
        html_dom_summary=_safe_dom(),
        html_dom_risk=0.1,
        layer2_capture={
            "input_registered_domain": "vercel.app",
            "final_registered_domain": "vercel.app",
            "final_url": "https://handshake-login.vercel.app/login",
            "contains_punycode": False,
            "contains_non_ascii": False,
            "brand_domain_mismatch": True,
            "final_domain_is_free_hosting": True,
        },
        legitimacy_bundle={**_safe_bundle(), "no_free_hosting_signal": False},
        cfg=_cfg(),
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["legitimacy_rescue_applied"] is False
    assert "cloud_hosted_brand_impersonation" in (out.get("legitimacy_rescue_blockers") or [])


def test_paypal_secure_netlify_app_detected_as_cloud_impersonation() -> None:
    assert _detect_cloud_hosted_brand_impersonation(
        final_registered_domain="netlify.app",
        final_host="paypal-secure.netlify.app",
        final_url="https://paypal-secure.netlify.app/auth",
    )


def test_microsoft_auth_github_io_detected_as_cloud_impersonation() -> None:
    assert _detect_cloud_hosted_brand_impersonation(
        final_registered_domain="github.io",
        final_host="microsoft-auth.github.io",
        final_url="https://microsoft-auth.github.io/login",
    )


def test_github_repo_page_not_auto_legit() -> None:
    trust = _evaluate_hosting_domain_trust(
        layer2_capture={
            "input_registered_domain": "github.com",
            "final_registered_domain": "github.com",
            "final_url": "https://github.com/user/repo",
        },
        html_structure_risk=0.05,
        html_dom_risk=0.08,
        blockers=[],
        cfg=_cfg(),
    )
    out = _apply_hosting_trust_promotion(
        _verdict_uncertain(),
        trust=trust,
        ml={"phish_proba": 0.55},
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["verdict_3way"] == "uncertain"
    assert out.get("hosting_trust_promotion_applied") is False


def test_unknown_clean_domain_remains_uncertain() -> None:
    trust = _evaluate_hosting_domain_trust(
        layer2_capture={
            "input_registered_domain": "example-unknown-site.com",
            "final_registered_domain": "example-unknown-site.com",
            "final_url": "https://example-unknown-site.com/login",
        },
        html_structure_risk=0.05,
        html_dom_risk=0.05,
        blockers=[],
        cfg=_cfg(),
    )
    out = _apply_hosting_trust_promotion(
        _verdict_uncertain(),
        trust=trust,
        ml={"phish_proba": 0.5},
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["verdict_3way"] == "uncertain"
    assert out.get("hosting_trust_promotion_applied") is False


def test_builder_hosting_404_page_can_downgrade_to_uncertain() -> None:
    verdict = {"combined_score": 0.79, "label": "likely_phishing", "verdict_3way": "likely_phishing", "reasons": []}
    out = _apply_untrusted_builder_hosting_downgrade(
        verdict,
        layer2_capture={
            "final_registered_domain": "framer.app",
            "final_url": "https://demo-404-project.framer.app/",
            "brand_domain_mismatch": False,
            "capture_failed": False,
            "title": "404 Not Found",
            "visible_text_sample": "This site is unavailable. Project not published.",
        },
        html_structure_summary={"form_count": 0, "password_input_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_dom_enrichment={"html_capture_missing_reason": None, "suspicious_html_keyword_count": 0},
        blockers=[],
    )
    assert out["verdict_3way"] == "uncertain"
    assert out["untrusted_builder_hosting_signal_applied"] is True
    assert "Untrusted hosting platform with generic hostname; legitimacy cannot be confirmed." in (out.get("reasons") or [])


def test_builder_hosting_framer_capture_failure_high_ml_not_downgraded() -> None:
    verdict = {"combined_score": 0.79, "label": "likely_phishing", "verdict_3way": "likely_phishing", "reasons": []}
    out = _apply_untrusted_builder_hosting_downgrade(
        verdict,
        layer2_capture={
            "final_registered_domain": "framer.app",
            "final_url": "https://valuable-population-385200.framer.app/",
            "brand_domain_mismatch": False,
            "capture_failed": True,
            "title": "",
            "visible_text_sample": "",
        },
        html_structure_summary={"form_count": 0, "password_input_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available", "suspicious_html_keyword_count": 0},
        blockers=[],
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert out["untrusted_builder_hosting_signal_applied"] is False


def test_builder_hosting_with_login_brand_payment_terms_not_downgraded() -> None:
    verdict = {"combined_score": 0.79, "label": "likely_phishing", "verdict_3way": "likely_phishing", "reasons": []}
    out = _apply_untrusted_builder_hosting_downgrade(
        verdict,
        layer2_capture={
            "final_registered_domain": "webflow.io",
            "final_url": "https://some-site.webflow.io/login",
            "brand_domain_mismatch": False,
            "capture_failed": False,
            "title": "Account Login",
            "visible_text_sample": "Sign in to verify your payment details for Microsoft account.",
        },
        html_structure_summary={"form_count": 1, "password_input_count": 1},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_dom_enrichment={"html_capture_missing_reason": None, "suspicious_html_keyword_count": 3},
        blockers=[],
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert out["untrusted_builder_hosting_signal_applied"] is False


def test_inactive_overlay_404_builder_host_marks_uncertain_with_tag() -> None:
    verdict = {"combined_score": 0.81, "label": "likely_phishing", "verdict_3way": "likely_phishing", "reasons": []}
    out = _apply_inactive_site_overlay(
        verdict,
        ml={"predicted_phishing": False, "phish_proba": 0.49},
        layer2_capture={
            "capture_failed": False,
            "title": "404 Not Found",
            "visible_text_sample": "Project not published.",
            "brand_domain_mismatch": False,
            "final_url": "https://site-404.framer.app/",
            "final_registered_domain": "framer.app",
        },
        html_structure_summary={"form_count": 0, "password_input_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_dom_enrichment={"html_capture_missing_reason": "html_parse_partial", "suspicious_html_keyword_count": 0},
        blockers=[],
    )
    assert out["verdict_3way"] == "uncertain"
    assert out["inactive_site_detected"] is True
    assert out["inactive_site_label"] == "inactive / no live content available"


def test_inactive_overlay_capture_failed_high_ml_suspicious_host_stays_phishing() -> None:
    verdict = {"combined_score": 0.86, "label": "likely_phishing", "verdict_3way": "likely_phishing", "reasons": []}
    out = _apply_inactive_site_overlay(
        verdict,
        ml={"predicted_phishing": True, "phish_proba": 0.91},
        layer2_capture={
            "capture_failed": True,
            "title": "",
            "visible_text_sample": "",
            "brand_domain_mismatch": False,
            "final_url": "https://valuable-population-385200.framer.app/",
            "final_registered_domain": "framer.app",
        },
        html_structure_summary={"form_count": 0, "password_input_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available", "suspicious_html_keyword_count": 0},
        blockers=["suspicious_host_pattern"],
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert out["inactive_site_detected"] is False


def test_inactive_overlay_real_phishing_temporarily_down_stays_phishing() -> None:
    verdict = {"combined_score": 0.79, "label": "likely_phishing", "verdict_3way": "likely_phishing", "reasons": []}
    out = _apply_inactive_site_overlay(
        verdict,
        ml={"predicted_phishing": True, "phish_proba": 0.84},
        layer2_capture={
            "capture_failed": True,
            "title": "Temporarily unavailable",
            "visible_text_sample": "",
            "brand_domain_mismatch": True,
            "final_url": "https://paypal-recovery-example.netlify.app/",
            "final_registered_domain": "netlify.app",
        },
        html_structure_summary={"form_count": 0, "password_input_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": True,
        },
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available", "suspicious_html_keyword_count": 0},
        blockers=["free_hosting_impersonation"],
    )
    assert out["verdict_3way"] == "likely_phishing"
    assert out["inactive_site_detected"] is False


def test_inactive_overlay_legit_site_down_marks_uncertain_with_tag() -> None:
    verdict = {"combined_score": 0.74, "label": "likely_phishing", "verdict_3way": "likely_phishing", "reasons": []}
    out = _apply_inactive_site_overlay(
        verdict,
        ml={"predicted_phishing": False, "phish_proba": 0.42},
        layer2_capture={
            "capture_failed": True,
            "title": "Site unavailable",
            "visible_text_sample": "The page is currently unavailable. Please try again later.",
            "brand_domain_mismatch": False,
            "final_url": "https://portal.example-edu.org/",
            "final_registered_domain": "example-edu.org",
        },
        html_structure_summary={"form_count": 0, "password_input_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available", "suspicious_html_keyword_count": 0},
        blockers=[],
    )
    assert out["verdict_3way"] == "uncertain"
    assert out["inactive_site_detected"] is True


def test_platform_context_official_weebly_login_detects_oauth() -> None:
    ctx = _classify_platform_host_context(
        layer2_capture={
            "final_url": "https://www.weebly.com/app/front-door/signin?path=login#/",
            "final_registered_domain": "weebly.com",
            "title": "Sign in - Weebly",
            "visible_text_sample": "Continue with Google Continue with Facebook Login with Square",
        },
        html_dom_enrichment={},
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        cfg=_cfg(),
    )
    assert ctx["platform_context_type"] == "official_platform_login"
    assert "google" in ctx["oauth_providers_detected"]
    assert "facebook" in ctx["oauth_providers_detected"]
    assert "square" in ctx["oauth_providers_detected"]


def test_platform_context_outlookkoo_weebly_is_impersonation_or_dormant() -> None:
    ctx = _classify_platform_host_context(
        layer2_capture={
            "final_url": "https://outlookkoo.weebly.com/",
            "final_registered_domain": "weebly.com",
            "title": "404 Not Found",
            "visible_text_sample": "site unavailable",
        },
        html_dom_enrichment={},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern"},
        cfg=_cfg(),
    )
    assert ctx["platform_context_type"] in {"cloud_hosted_brand_impersonation", "user_hosted_subdomain"}
    assert "dormant_phishing_infra" in (ctx.get("platform_context_blockers") or [])
    # Policy should never convert this to likely_legitimate.
    out = _apply_platform_context_policy(
        _verdict_uncertain(),
        platform_context={**ctx, "dormant_phishing_infra_detected": True, "dormant_phishing_infra_reasons": ["test"]},
        html_structure_summary={"password_input_count": 0},
        html_dom_summary=_safe_dom(),
        html_structure_risk=0.1,
        html_dom_risk=0.1,
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["verdict_3way"] != "likely_legitimate"


def test_platform_context_official_framer_domain_detected() -> None:
    ctx = _classify_platform_host_context(
        layer2_capture={
            "final_url": "https://www.framer.com/",
            "final_registered_domain": "framer.com",
            "title": "Framer",
            "visible_text_sample": "Build websites",
        },
        html_dom_enrichment={},
        host_path_reasoning={},
        cfg=_cfg(),
    )
    assert ctx["platform_context_type"] == "official_platform_domain"


def test_platform_context_random_framer_app_is_user_hosted_not_legit() -> None:
    ctx = _classify_platform_host_context(
        layer2_capture={
            "final_url": "https://obvious-event-989161.framer.app/",
            "final_registered_domain": "framer.app",
            "title": "Page not found",
            "visible_text_sample": "404 page not found",
        },
        html_dom_enrichment={},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern"},
        cfg=_cfg(),
    )
    assert ctx["platform_context_type"] == "user_hosted_subdomain"
    out = _apply_platform_context_policy(
        _verdict_phish(),
        platform_context={
            **ctx,
            "dormant_phishing_infra_detected": True,
            "dormant_phishing_infra_reasons": ["Inactive page on suspicious user-hosted infrastructure."],
        },
        html_structure_summary={"password_input_count": 0},
        html_dom_summary=_safe_dom(),
        html_structure_risk=0.1,
        html_dom_risk=0.1,
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_platform_context_paypal_vercel_is_cloud_impersonation() -> None:
    ctx = _classify_platform_host_context(
        layer2_capture={
            "final_url": "https://paypal-login.vercel.app",
            "final_registered_domain": "vercel.app",
            "title": "PayPal verify",
            "visible_text_sample": "login",
        },
        html_dom_enrichment={},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern"},
        cfg=_cfg(),
    )
    assert ctx["platform_context_type"] == "cloud_hosted_brand_impersonation"
    out = _apply_platform_context_policy(
        _verdict_uncertain(),
        platform_context=ctx,
        html_structure_summary={"password_input_count": 1},
        html_dom_summary=_safe_dom(),
        html_structure_risk=0.2,
        html_dom_risk=0.2,
        verdict_cfg=Verdict3WayConfig(),
    )
    assert out["verdict_3way"] == "likely_phishing"


def test_platform_context_microsoft_github_pages_is_cloud_impersonation() -> None:
    ctx = _classify_platform_host_context(
        layer2_capture={
            "final_url": "https://microsoft-auth.github.io",
            "final_registered_domain": "github.io",
            "title": "Microsoft account",
            "visible_text_sample": "sign in",
        },
        html_dom_enrichment={},
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern"},
        cfg=_cfg(),
    )
    assert ctx["platform_context_type"] == "cloud_hosted_brand_impersonation"


def test_platform_context_unknown_inactive_page_stays_uncertain_with_inactive_tag() -> None:
    base = _apply_platform_context_policy(
        _verdict_uncertain(),
        platform_context={
            "platform_context_type": "unknown",
            "platform_name": None,
            "platform_context_reasons": [],
            "oauth_providers_detected": [],
            "platform_context_blockers": [],
        },
        html_structure_summary={"form_count": 0, "password_input_count": 0},
        html_dom_summary=_safe_dom(),
        html_structure_risk=0.1,
        html_dom_risk=0.1,
        verdict_cfg=Verdict3WayConfig(),
    )
    out = _apply_inactive_site_overlay(
        base,
        ml={"predicted_phishing": False, "phish_proba": 0.45},
        layer2_capture={
            "capture_failed": True,
            "title": "404 Not Found",
            "visible_text_sample": "Site unavailable",
            "brand_domain_mismatch": False,
            "final_url": "https://unknown-example-site123.com/",
            "final_registered_domain": "unknown-example-site123.com",
        },
        html_structure_summary={"form_count": 0, "password_input_count": 0},
        html_dom_summary={
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_dom_enrichment={"html_capture_missing_reason": "html_not_available", "suspicious_html_keyword_count": 0},
        blockers=[],
    )
    assert out["verdict_3way"] == "uncertain"
    assert out["inactive_site_detected"] is True
