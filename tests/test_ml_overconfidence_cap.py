from src.app_v1.analyze_dashboard import _apply_ml_overconfidence_cap


def _base_cap() -> dict:
    return {
        "input_registered_domain": "framer.com",
        "final_registered_domain": "framer.com",
        "final_url": "https://www.framer.com/pricing",
        "brand_domain_mismatch": True,
        "brand_domain_mismatch_strength": "weak",
    }


def test_official_content_page_applies_overconfidence_cap() -> None:
    score, meta = _apply_ml_overconfidence_cap(
        ml_effective_score=0.99,
        layer2_capture=_base_cap(),
        html_structure_summary={"password_input_count": 0},
        html_dom_summary={
            "page_family": "generic_landing",
            "content_rich_profile": True,
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_structure_risk=0.20,
        html_dom_risk=0.10,
        host_path_reasoning={"host_legitimacy_confidence": "high", "host_identity_class": "known_platform_official"},
        platform_context={"platform_context_type": "official_platform_domain"},
        hosting_trust={"hosting_trust_status": "hosting_trust_partial"},
    )
    assert score is not None and score <= 0.60
    assert meta["ml_overconfidence_cap_applied"] is True


def test_user_hosted_subdomain_does_not_apply_cap() -> None:
    cap = _base_cap()
    cap["final_registered_domain"] = "framer.app"
    cap["input_registered_domain"] = "framer.app"
    cap["final_url"] = "https://evil-123.framer.app/"
    score, meta = _apply_ml_overconfidence_cap(
        ml_effective_score=0.95,
        layer2_capture=cap,
        html_structure_summary={"password_input_count": 0},
        html_dom_summary={
            "page_family": "generic_landing",
            "content_rich_profile": True,
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_structure_risk=0.20,
        html_dom_risk=0.10,
        host_path_reasoning={"host_legitimacy_confidence": "medium", "host_identity_class": "suspicious_host_pattern"},
        platform_context={"platform_context_type": "user_hosted_subdomain"},
        hosting_trust={"hosting_trust_status": "hosting_trust_unknown"},
    )
    assert score == 0.95
    assert meta["ml_overconfidence_cap_applied"] is False


def test_cloud_impersonation_does_not_apply_cap() -> None:
    score, meta = _apply_ml_overconfidence_cap(
        ml_effective_score=0.95,
        layer2_capture={
            "input_registered_domain": "vercel.app",
            "final_registered_domain": "vercel.app",
            "final_url": "https://paypal-login.vercel.app/",
            "brand_domain_mismatch": True,
            "brand_domain_mismatch_strength": "strong",
        },
        html_structure_summary={"password_input_count": 1},
        html_dom_summary={
            "page_family": "auth_login_recovery",
            "content_rich_profile": False,
            "form_action_external_domain_count": 1,
            "suspicious_credential_collection_pattern": True,
            "login_harvester_pattern": True,
            "trust_surface_brand_domain_mismatch": True,
        },
        html_structure_risk=0.80,
        html_dom_risk=0.75,
        host_path_reasoning={"host_legitimacy_confidence": "low", "host_identity_class": "suspicious_host_pattern"},
        platform_context={"platform_context_type": "cloud_hosted_brand_impersonation"},
        hosting_trust={"hosting_trust_status": "hosting_trust_unknown"},
    )
    assert score == 0.95
    assert meta["ml_overconfidence_cap_applied"] is False


def test_login_with_credentials_does_not_apply_cap() -> None:
    score, meta = _apply_ml_overconfidence_cap(
        ml_effective_score=0.91,
        layer2_capture={
            "input_registered_domain": "amazon.com",
            "final_registered_domain": "amazon.com",
            "final_url": "https://www.amazon.com/ap/signin",
            "brand_domain_mismatch": False,
            "brand_domain_mismatch_strength": "none",
        },
        html_structure_summary={"password_input_count": 1},
        html_dom_summary={
            "page_family": "auth_login_recovery",
            "content_rich_profile": False,
            "form_action_external_domain_count": 0,
            "suspicious_credential_collection_pattern": False,
            "login_harvester_pattern": False,
            "trust_surface_brand_domain_mismatch": False,
        },
        html_structure_risk=0.22,
        html_dom_risk=0.12,
        host_path_reasoning={"host_legitimacy_confidence": "high", "host_identity_class": "known_brand_official"},
        platform_context={"platform_context_type": "official_platform_login"},
        hosting_trust={"hosting_trust_status": "hosting_trust_verified"},
    )
    assert score == 0.91
    assert meta["ml_overconfidence_cap_applied"] is False
