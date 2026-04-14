from src.app_v1.analyze_dashboard import _apply_legitimacy_rescue_on_verdict


def test_legitimacy_rescue_applies_for_high_legit_content() -> None:
    verdict = {"combined_score": 0.61, "reasons": []}
    hp = {
        "host_identity_class": "public_content_platform",
        "host_legitimacy_confidence": "high",
        "path_fit_assessment": "plausible",
    }
    dom = {
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
    bundle = {
        "suspicious_form_action_cross_origin": False,
        "no_deceptive_token_placement": True,
        "no_free_hosting_signal": True,
    }
    out = _apply_legitimacy_rescue_on_verdict(
        verdict,
        host_path_reasoning=hp,
        html_dom_summary=dom,
        html_dom_risk=0.14,
        legitimacy_bundle=bundle,
    )
    assert out["legitimacy_rescue_applied"] is True
    assert out["combined_score"] < 0.61


def test_legitimacy_rescue_blocked_for_phishing_context() -> None:
    verdict = {"combined_score": 0.61, "reasons": []}
    hp = {
        "host_identity_class": "public_content_platform",
        "host_legitimacy_confidence": "high",
        "path_fit_assessment": "plausible",
    }
    dom = {
        "page_family": "article_news",
        "trust_action_context": True,
        "form_action_external_domain_count": 1,
    }
    bundle = {
        "suspicious_form_action_cross_origin": True,
        "no_deceptive_token_placement": True,
        "no_free_hosting_signal": True,
    }
    out = _apply_legitimacy_rescue_on_verdict(
        verdict,
        host_path_reasoning=hp,
        html_dom_summary=dom,
        html_dom_risk=0.44,
        legitimacy_bundle=bundle,
    )
    assert out["legitimacy_rescue_applied"] is False
    assert out["combined_score"] == 0.61
