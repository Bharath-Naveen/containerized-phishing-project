"""Structural brand / impersonation features and label-probability mapping."""

import numpy as np
import pytest

from src.pipeline.features.brand_signals import BRAND_TOKENS, host_on_official_brand_apex
from src.pipeline.label_policy import phish_probability_from_proba_row
from src.pipeline.layer1_features import extract_layer1_features


def test_official_google_and_accounts() -> None:
    for u in ("https://google.com", "https://accounts.google.com"):
        row = extract_layer1_features(u, use_dns=False)
        assert row["official_registrable_anchor"] == 1
        assert row["brand_hostname_exact_label_match"] == 1
        assert row["brand_hyphenated_deception_label"] == 0
        assert row["layer1_brand_trust_score"] == 3


def test_github_stripe_official_domain_family() -> None:
    for u in ("https://github.com/login", "https://stripe.com/docs"):
        row = extract_layer1_features(u, use_dns=False)
        assert row["official_registrable_anchor"] == 1, u
        assert row["official_domain_family"] == 1, u


def test_paypal_amazon_microsoftonline() -> None:
    cases = [
        ("https://paypal.com", "paypal.com"),
        ("https://amazon.com", "amazon.com"),
        ("https://login.microsoftonline.com", "microsoftonline.com"),
    ]
    for url, _ in cases:
        row = extract_layer1_features(url, use_dns=False)
        assert row["official_registrable_anchor"] == 1, url


def test_hyphenated_brand_deception() -> None:
    row = extract_layer1_features("https://google-login-secure.xyz/path", use_dns=False)
    assert row["brand_hyphenated_deception_label"] == 1
    assert row["official_registrable_anchor"] == 0


def test_brand_on_free_hosting() -> None:
    row = extract_layer1_features(
        "https://paypal-verification-help.netlify.app/login",
        use_dns=False,
    )
    assert row["free_hosting_flag"] == 1
    assert row["brand_on_free_hosting"] == 1
    assert row["official_registrable_anchor"] == 0


def test_path_brand_without_official_host() -> None:
    row = extract_layer1_features("https://evil.co/google/signin", use_dns=False)
    assert row["path_brand_token_present"] == 1
    assert row["path_brand_without_official_host"] == 1


def test_malformed_url_safe_layer1() -> None:
    bad = "http://[::1"
    row = extract_layer1_features(bad, use_dns=False)
    assert row["hosting_features_missing"] == 1
    assert row["official_registrable_anchor"] == 0


def test_proba_mapping_uses_classes_order() -> None:
    classes = np.array([0, 1])
    row = np.array([0.85, 0.15])
    assert phish_probability_from_proba_row(row, classes) == pytest.approx(0.15)
    classes2 = np.array([1, 0])
    row2 = np.array([0.2, 0.8])
    assert phish_probability_from_proba_row(row2, classes2) == pytest.approx(0.2)


def test_host_on_official_apex_subdomain() -> None:
    assert host_on_official_brand_apex("pay.google.com") is True
    assert host_on_official_brand_apex("google.com.evil.net") is False


def test_typosquat_prefix_not_meta_in_microsoft() -> None:
    row = extract_layer1_features("https://microsoft.com", use_dns=False)
    assert row["brand_typosquat_embedded_in_label"] == 0


def test_brand_tokens_nonempty() -> None:
    assert len(BRAND_TOKENS) >= 8
