"""Regression tests for the audit findings fixed in rebuild Phase 1."""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.guards import PipelineGuardError, check_feature_parity, check_group_keys, check_no_group_overlap
from src.pipeline.layer1_features import extract_layer1_features
from src.pipeline.safe_url import leak_safe_group_key
from src.pipeline.train import LAYER1_EXCLUDE_FROM_X
from src.pipeline.url_normalize import canonical_url, feature_url

SAME_ADDRESS = [
    "summah.info/",
    "http://summah.info/",
    "https://summah.info/",
    "HTTPS://Summah.info",
    "  summah.info  ",
]


def _model_features(row):
    return {k: v for k, v in row.items() if k not in LAYER1_EXCLUDE_FROM_X and k != "canonical_url"}


def test_bare_host_gets_real_group_key():
    # The shipped models' split broke because this returned malformed::<hash>.
    assert leak_safe_group_key("summah.info/")[0] == "summah.info"
    assert leak_safe_group_key("jobspert.com/employer/x/")[0] == "jobspert.com"


def test_same_address_same_model_features_regardless_of_scheme_or_form():
    feats = [_model_features(extract_layer1_features(u)) for u in SAME_ADDRESS]
    for f in feats[1:]:
        assert f == feats[0]


def test_bare_host_has_hostname_features():
    f = extract_layer1_features("guitars101.com/forums/f90/page.html")
    assert f["hostname_length"] == len("guitars101.com")
    assert f["hosting_features_missing"] == 0


def test_scheme_derived_columns_never_trained_on():
    assert {"has_https", "https_without_official_anchor", "official_anchor_with_https"} <= LAYER1_EXCLUDE_FROM_X


def test_has_https_still_reports_original_scheme_for_display():
    assert extract_layer1_features("https://example.com")["has_https"] == 1
    assert extract_layer1_features("example.com")["has_https"] == 0


def test_feature_url_is_scheme_neutral_and_canonical():
    assert feature_url("https://Example.com/a/") == "http://example.com/a"
    assert canonical_url("example.com")[0] == "http://example.com/"


def test_group_guard_trips_on_mass_malformed_keys():
    keys = ["malformed::x%d" % i for i in range(50)] + ["a.com"] * 50
    with pytest.raises(PipelineGuardError):
        check_group_keys(keys, where="test")


def test_overlap_guard_trips():
    with pytest.raises(PipelineGuardError):
        check_no_group_overlap(["a.com", "b.com"], ["b.com"], where="test")


def test_parity_guard_detects_skew():
    good = extract_layer1_features("http://example.com/login")
    df = pd.DataFrame([{**good}])
    assert check_feature_parity(df, ["hostname_length", "url_length"], n=1)["mismatched_rows"] == 0
    bad = pd.DataFrame([{**good, "hostname_length": 0}])
    with pytest.raises(PipelineGuardError):
        check_feature_parity(bad, ["hostname_length"], n=1)


def test_evaluation_urls_are_excluded_from_training():
    from src.pipeline.evaluation_sets import drop_evaluation_rows, load_url_suites

    urls = [u for v in load_url_suites().values() for u in v] + ["http://some-unrelated-site.org/x"]
    df = pd.DataFrame({"canonical_url": [canonical_url(u)[0] for u in urls]})
    kept, stats = drop_evaluation_rows(df)
    assert list(kept["canonical_url"]) == ["http://some-unrelated-site.org/x"]
    assert stats["rows_out"] == 1


def test_obvious_phishing_is_not_softened_to_uncertain_without_capture():
    """Audit repro: raw ML ~0.97 on these, but the EAL returned 'uncertain'."""
    from src.app_v1.host_path_reasoning import assess_host_path_reasoning

    for u in ("https://google-login-secure.xyz/signin", "https://totally-fake-bank-login.xyz/verify"):
        hp = assess_host_path_reasoning(input_url=u)["host_path_reasoning"]
        assert hp["host_identity_class"] == "suspicious_host_pattern", u


def test_official_and_ordinary_hosts_not_flagged_by_new_host_rules():
    from src.app_v1.host_path_reasoning import assess_host_path_reasoning

    for u in ("https://accounts.google.com/signin", "https://www.paypal.com/signin", "https://github.com/login",
              "https://www.wikipedia.org/", "https://secure.bankofamerica.com/login", "https://mybank-online.com/"):
        hp = assess_host_path_reasoning(input_url=u)["host_path_reasoning"]
        assert hp["host_identity_class"] != "suspicious_host_pattern", u
