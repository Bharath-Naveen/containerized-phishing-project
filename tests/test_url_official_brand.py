"""Official-domain policy helper + dashboard cap (fallback UX)."""

from src.app_v1.analyze_dashboard import _apply_official_brand_apex_cap
from src.pipeline.features.brand_signals import host_on_official_brand_apex
from src.pipeline.features.url_features import host_on_official_brand_apex as host_on_official_reexport
from src.pipeline.layer1_features import extract_layer1_features


def test_layer1_google_official_anchor() -> None:
    row = extract_layer1_features("https://google.com", use_dns=False)
    assert row["official_registrable_anchor"] == 1
    assert row["brand_hostname_exact_label_match"] == 1


def test_typosquat_still_suspicious() -> None:
    row = extract_layer1_features("https://google.com.evil.net", use_dns=False)
    assert row["official_registrable_anchor"] == 0


def test_host_on_official_apex() -> None:
    assert host_on_official_brand_apex("www.google.com") is True
    assert host_on_official_brand_apex("pay.google.com") is True
    assert host_on_official_brand_apex("google.com") is True


def test_url_features_reexports_host_check() -> None:
    assert host_on_official_reexport is host_on_official_brand_apex


def test_dashboard_cap_lowers_phish_for_official_apex() -> None:
    ml = {
        "phish_proba": 0.99,
        "canonical_url": "https://google.com",
        "predicted_phishing": True,
    }
    out = _apply_official_brand_apex_cap(ml, "https://google.com")
    assert out["phish_proba"] <= 0.32
    assert out["predicted_phishing"] is False
    assert out.get("phish_proba_model_raw") == 0.99
