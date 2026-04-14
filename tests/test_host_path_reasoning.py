from src.app_v1.host_path_reasoning import (
    assess_host_path_reasoning,
    blend_ml_phish_for_host_path_reasoning,
)


def test_public_content_path_plausible_reddit_style() -> None:
    out = assess_host_path_reasoning(
        input_url="https://www.reddit.com/r/news",
        html_dom_summary={
            "page_family": "content_feed_forum_aggregator",
            "trust_action_context": False,
            "form_action_external_domain_count": 0,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": False,
            "interstitial_or_preview_pattern": False,
            "anchor_strong_mismatch_count": 0,
            "suspicious_credential_collection_pattern": False,
        },
        legitimacy_bundle={"no_deceptive_token_placement": True, "no_free_hosting_signal": True},
    )
    hp = out["host_path_reasoning"]
    assert hp is not None
    assert hp["host_legitimacy_confidence"] in {"high", "medium"}
    assert hp["path_fit_assessment"] == "plausible"
    assert hp["host_identity_class"] in {"social_forum_feed", "public_content_platform"}


def test_public_auth_path_plausible_github_login() -> None:
    out = assess_host_path_reasoning(
        input_url="https://github.com/login",
        html_dom_summary={"page_family": "auth_login_recovery", "trust_action_context": True},
        legitimacy_bundle={"no_deceptive_token_placement": True, "no_free_hosting_signal": True},
    )
    hp = out["host_path_reasoning"]
    assert hp is not None
    assert hp["path_fit_assessment"] in {"plausible", "unusual_but_possible"}
    assert hp["host_legitimacy_confidence"] in {"high", "medium"}


def test_host_userinfo_confusion_marked_suspicious() -> None:
    out = assess_host_path_reasoning(
        input_url="https://reddit.com@evil.xyz/login",
        html_dom_summary={"page_family": "auth_login_recovery", "trust_action_context": True},
    )
    hp = out["host_path_reasoning"]
    assert hp is not None
    assert hp["host_identity_class"] == "suspicious_host_pattern"
    assert hp["path_fit_assessment"] == "suspicious"


def test_content_host_with_verify_path_no_trust_context_is_suspicious() -> None:
    out = assess_host_path_reasoning(
        input_url="https://example-forum.com/verify-account-paypal",
        html_dom_summary={
            "page_family": "content_feed_forum_aggregator",
            "trust_action_context": False,
            "form_action_external_domain_count": 0,
            "login_harvester_pattern": False,
            "wrapper_page_pattern": False,
            "interstitial_or_preview_pattern": False,
            "anchor_strong_mismatch_count": 0,
            "suspicious_credential_collection_pattern": False,
        },
    )
    hp = out["host_path_reasoning"]
    assert hp is not None
    assert hp["path_fit_assessment"] == "suspicious"


def test_host_path_discount_applies_only_when_safe() -> None:
    hp = {
        "host_identity_class": "public_content_platform",
        "host_legitimacy_confidence": "high",
        "path_fit_assessment": "plausible",
    }
    dom = {
        "trust_action_context": False,
        "form_action_external_domain_count": 0,
        "login_harvester_pattern": False,
        "wrapper_page_pattern": False,
        "interstitial_or_preview_pattern": False,
        "anchor_strong_mismatch_count": 0,
        "suspicious_credential_collection_pattern": False,
    }
    p2, meta = blend_ml_phish_for_host_path_reasoning(
        phish_proba=0.62,
        host_path_reasoning=hp,
        html_dom_summary=dom,
        legitimacy_bundle={
            "suspicious_form_action_cross_origin": False,
            "no_deceptive_token_placement": True,
            "no_free_hosting_signal": True,
        },
    )
    assert p2 is not None and p2 < 0.62
    assert meta["host_path_ml_discount_applied"] is True

    p3, meta2 = blend_ml_phish_for_host_path_reasoning(
        phish_proba=0.62,
        host_path_reasoning=hp,
        html_dom_summary={**dom, "trust_action_context": True},
        legitimacy_bundle={
            "suspicious_form_action_cross_origin": False,
            "no_deceptive_token_placement": True,
            "no_free_hosting_signal": True,
        },
    )
    assert p3 == 0.62
    assert meta2["host_path_ml_discount_applied"] is False
