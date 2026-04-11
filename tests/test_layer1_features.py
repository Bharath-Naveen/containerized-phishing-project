"""Layer-1 feature extraction shape and keys."""

from src.pipeline.layer1_features import extract_layer1_features, layer1_feature_key_set


def test_layer1_keys_stable_without_dns() -> None:
    keys = layer1_feature_key_set(include_dns=False)
    assert "url_length" in keys
    assert "domain_hash_bucket" in keys
    assert "canonical_url" not in keys


def test_extract_layer1_no_dns() -> None:
    row = extract_layer1_features("https://sub.example.com/path?q=1", use_dns=False)
    assert row["dns_features_skipped"] == 1
    assert "registered_domain" not in row
    assert isinstance(row["domain_hash_bucket"], int)
