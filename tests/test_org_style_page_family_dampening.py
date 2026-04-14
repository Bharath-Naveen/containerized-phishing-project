from src.app_v1.org_style_signals import dampen_org_style_for_page_family


def test_content_page_brand_mentions_are_dampened_without_trust_context() -> None:
    org = {
        "org_style_risk_score": 0.48,
        "brand_domain_mismatches": ["google", "apple"],
        "reasons": ["Content references brand tokens."],
    }
    dom = {
        "page_family": "article_news",
        "trust_action_context": False,
        "form_action_external_domain_count": 0,
        "login_harvester_pattern": False,
        "wrapper_page_pattern": False,
        "interstitial_or_preview_pattern": False,
        "trust_surface_brand_domain_mismatch": False,
        "anchor_strong_mismatch_count": 0,
        "suspicious_credential_collection_pattern": False,
    }
    out = dampen_org_style_for_page_family(org, dom)
    assert out["org_style_risk_score"] < 0.30
    assert any("de-emphasized" in r for r in (out.get("reasons") or []))


def test_dampening_not_applied_with_strong_phish_context() -> None:
    org = {
        "org_style_risk_score": 0.48,
        "brand_domain_mismatches": ["google"],
        "reasons": [],
    }
    dom = {
        "page_family": "article_news",
        "trust_action_context": True,
        "form_action_external_domain_count": 1,
    }
    out = dampen_org_style_for_page_family(org, dom)
    assert out["org_style_risk_score"] == 0.48
