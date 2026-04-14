"""Malformed URLs (e.g. invalid IPv6 netloc) must not crash clean, features, or leak-safe split."""

from __future__ import annotations

import pytest
from urllib.parse import urlsplit

# Triggers urllib ValueError: Invalid IPv6 URL (unclosed bracket / bad literal).
BAD_IPV6_NETLOC = "http://[::1"


def test_urlsplit_raises_on_known_bad_input() -> None:
    with pytest.raises(ValueError, match="Invalid IPv6"):
        urlsplit(BAD_IPV6_NETLOC)


def test_safe_urlsplit_returns_error_not_exception() -> None:
    from src.pipeline.safe_url import safe_urlsplit

    parts, err = safe_urlsplit(BAD_IPV6_NETLOC)
    assert parts is None
    assert err is not None
    assert "IPv6" in err or "ipv6" in err.lower()


def test_leak_safe_group_key_fallback() -> None:
    from src.pipeline.safe_url import leak_safe_group_key

    key, fb = leak_safe_group_key(BAD_IPV6_NETLOC)
    assert fb == 1
    assert key.startswith("malformed::")
    assert len(key) > len("malformed::")


def test_leak_safe_group_key_stable() -> None:
    from src.pipeline.safe_url import leak_safe_group_key

    k1, _ = leak_safe_group_key(BAD_IPV6_NETLOC)
    k2, _ = leak_safe_group_key(BAD_IPV6_NETLOC)
    assert k1 == k2


def test_extract_url_features_no_raise() -> None:
    from src.pipeline.features.url_features import extract_url_features

    out = extract_url_features(BAD_IPV6_NETLOC)
    assert out["hostname_length"] == 0
    assert out["url_length"] > 0


def test_extract_hosting_features_no_raise() -> None:
    from src.pipeline.features.hosting_features import extract_hosting_features

    out = extract_hosting_features(BAD_IPV6_NETLOC)
    assert out["registered_domain"] == ""
    assert out["private_host_flag"] == 0


def test_extract_layer1_features_no_raise() -> None:
    from src.pipeline.layer1_features import extract_layer1_features

    row = extract_layer1_features(BAD_IPV6_NETLOC, use_dns=False)
    assert row["hosting_features_missing"] == 1
    assert row["dns_features_skipped"] == 1


def test_stratified_group_split_with_malformed_url() -> None:
    import pandas as pd

    from src.pipeline.split_leak_safe import stratified_group_train_test

    n = 40
    urls = [f"https://phish{k}.campaign.example/p" for k in range(n // 2)] + [
        f"https://legit{k}.safe.example/p" for k in range(n // 2)
    ]
    labels = [1] * (n // 2) + [0] * (n // 2)
    urls[7] = BAD_IPV6_NETLOC
    df = pd.DataFrame({"canonical_url": urls, "label": labels})

    train, test, _st = stratified_group_train_test(df, test_size=0.2, random_state=42)

    assert "group_key_fallback_used" in train.columns
    assert "group_key_fallback_used" in test.columns
    combined = pd.concat([train, test], ignore_index=True)
    assert (combined["group_key_fallback_used"].astype(int) == 1).any()
