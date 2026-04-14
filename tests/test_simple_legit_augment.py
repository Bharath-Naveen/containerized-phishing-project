"""Curated simple-legit rows merged into cleaned frames."""

import pandas as pd

from src.pipeline.clean import canonicalize_url
from src.pipeline.simple_legit_augment import augment_cleaned_with_simple_legit, curated_legit_augment_rows


def test_curated_legit_merges_simple_and_hard_jsonl() -> None:
    rows = curated_legit_augment_rows(include_hard_legit=True)
    assert len(rows) >= 500
    urls = " ".join(r["url"].lower() for r in rows)
    assert "microsoftonline" in urls or "paypal.com" in urls


def test_augment_adds_rows_and_skips_duplicates() -> None:
    base = pd.DataFrame(
        {
            "url": ["https://www.example.com/a"],
            "label": [0],
        }
    )
    c, inv, err = canonicalize_url("https://www.example.com/a")
    base["canonical_url"] = [c]
    base["invalid_url"] = [inv]
    base["parse_error"] = [err]
    base["source_file"] = ["x"]
    base["source_dataset"] = ["y"]
    base["source_brand_hint"] = [""]
    base["action_category"] = ["kaggle_primary"]
    base["kaggle_raw_status"] = ["1"]

    out, stats = augment_cleaned_with_simple_legit(base)
    assert stats["rows_added"] >= 1
    assert len(out) == len(base) + stats["rows_added"]

    out2, stats2 = augment_cleaned_with_simple_legit(out)
    assert stats2["rows_added"] == 0
