"""Stratified sampling preserves both labels when possible."""

import pandas as pd

from src.pipeline.stratified_sample import stratified_sample_by_label


def test_stratified_sample_n_preserves_both_classes() -> None:
    df = pd.DataFrame(
        {
            "url": [f"https://a{i}.com" for i in range(20)]
            + [f"https://b{i}.evil" for i in range(20)],
            "label": [0] * 20 + [1] * 20,
        }
    )
    out, stats = stratified_sample_by_label(df, n=10, random_state=0)
    assert len(out) == 10
    assert stats["sample_rows"] == 10
    assert set(out["label"].unique()) == {0, 1}


def test_stratified_sample_frac() -> None:
    df = pd.DataFrame({"url": ["https://x.com"] * 100, "label": [0] * 50 + [1] * 50})
    out, _ = stratified_sample_by_label(df, frac=0.2, random_state=1)
    assert len(out) == 20
