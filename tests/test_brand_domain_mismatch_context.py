from src.app_v1.analyze_dashboard import _compute_phishing_blockers, _enrich_capture_and_html_signals


def _safe_bundle() -> dict:
    return {
        "suspicious_form_action_cross_origin": False,
        "no_deceptive_token_placement": True,
        "no_free_hosting_signal": True,
    }


def test_pricing_page_payment_mention_is_weak_mismatch_not_blocker() -> None:
    layer2, _, _, _ = _enrich_capture_and_html_signals(
        input_url="https://www.framer.com/pricing",
        capture_json={
            "final_url": "https://www.framer.com/pricing",
            "redirect_chain": [],
            "cross_domain_redirect_count": 0,
        },
        soup=None,
        html_structure_summary={
            "brand_terms_found_in_text": ["paypal", "stripe"],
            "password_input_count": 0,
            "email_input_count": 0,
            "phone_input_count": 0,
            "form_count": 0,
            "visible_text_snippet": "Pay with PayPal. Supports Stripe and cards.",
            "title": "Framer Pricing",
        },
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "generic_landing"},
    )
    assert layer2["brand_domain_mismatch"] is True
    assert layer2["brand_domain_mismatch_strength"] == "weak"
    blockers = _compute_phishing_blockers(
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        html_dom_summary={"form_action_external_domain_count": 0, "trust_surface_brand_domain_mismatch": False},
        layer2_capture=layer2,
        legitimacy_bundle=_safe_bundle(),
        cloud_hosted_brand_impersonation=False,
    )
    assert "brand_domain_mismatch" not in blockers


def test_login_page_brand_mismatch_is_strong_blocker() -> None:
    layer2, _, _, _ = _enrich_capture_and_html_signals(
        input_url="https://www.framer.com/login",
        capture_json={
            "final_url": "https://www.framer.com/login",
            "redirect_chain": [],
            "cross_domain_redirect_count": 0,
        },
        soup=None,
        html_structure_summary={
            "brand_terms_found_in_text": ["paypal"],
            "password_input_count": 1,
            "email_input_count": 1,
            "phone_input_count": 0,
            "form_count": 1,
            "visible_text_snippet": "PayPal account login required.",
            "title": "Login",
        },
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "auth_login_recovery"},
    )
    assert layer2["brand_domain_mismatch"] is True
    assert layer2["brand_domain_mismatch_strength"] == "strong"
    blockers = _compute_phishing_blockers(
        host_path_reasoning={"host_identity_class": "known_platform_official"},
        html_dom_summary={"form_action_external_domain_count": 0, "trust_surface_brand_domain_mismatch": False},
        layer2_capture=layer2,
        legitimacy_bundle=_safe_bundle(),
        cloud_hosted_brand_impersonation=False,
    )
    assert "brand_domain_mismatch" in blockers


def test_cloud_impersonation_stays_strong_even_with_brand_mismatch_logic() -> None:
    layer2, _, _, _ = _enrich_capture_and_html_signals(
        input_url="https://paypal-login.vercel.app/login",
        capture_json={
            "final_url": "https://paypal-login.vercel.app/login",
            "redirect_chain": [],
            "cross_domain_redirect_count": 0,
        },
        soup=None,
        html_structure_summary={
            "brand_terms_found_in_text": ["paypal"],
            "password_input_count": 1,
            "email_input_count": 1,
            "phone_input_count": 0,
            "form_count": 1,
            "visible_text_snippet": "PayPal sign in",
            "title": "PayPal Login",
        },
        html_dom_summary={"form_action_external_domain_count": 0, "page_family": "auth_login_recovery"},
    )
    assert layer2["brand_domain_mismatch_strength"] == "strong"
    blockers = _compute_phishing_blockers(
        host_path_reasoning={"host_identity_class": "suspicious_host_pattern"},
        html_dom_summary={"form_action_external_domain_count": 0, "trust_surface_brand_domain_mismatch": False},
        layer2_capture=layer2,
        legitimacy_bundle={**_safe_bundle(), "no_free_hosting_signal": False},
        cloud_hosted_brand_impersonation=True,
    )
    assert "brand_domain_mismatch" in blockers
    assert "cloud_hosted_brand_impersonation" in blockers
