"""Simple-legitimate surface URL features (path depth, auth tokens, redirect query keys)."""

from src.pipeline.features.url_features import extract_url_features
from src.pipeline.layer1_features import extract_layer1_features


def test_homepage_shallow_path_google() -> None:
    f = extract_url_features("https://www.google.com/")
    assert f["path_segment_count"] == 0 or f["path_shallow_le1"] == 1
    assert f["no_authish_path_query_tokens"] == 1
    assert f["suspicious_redirect_query_flag"] == 0


def test_login_path_has_authish_hits() -> None:
    f = extract_url_features("https://evil.com/account/login")
    assert f["path_query_authish_keyword_hits"] >= 1
    assert f["no_authish_path_query_tokens"] == 0


def test_redirect_query_flag() -> None:
    f = extract_url_features("https://evil.com/x?redirect=https://bank.com&next=/ok")
    assert f["suspicious_redirect_query_flag"] == 1


def test_layer1_simple_official_homepage_shape() -> None:
    row = extract_layer1_features("https://www.amazon.com/", use_dns=False)
    assert row.get("simple_official_homepage_shape") == 1
    assert row.get("official_registrable_anchor") == 1
