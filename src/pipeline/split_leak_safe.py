"""Stratified train/test split with group holdout (registered_domain) to limit campaign leakage."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.pipeline.paths import ensure_layout, processed_dir, reports_dir
from src.pipeline.safe_url import leak_safe_group_key

logger = logging.getLogger(__name__)


def _leak_group_and_fallback(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Per-row registered-domain style group key; malformed URLs get deterministic ``malformed::<hash>``."""
    urls = df["canonical_url"].fillna("").astype(str)
    keys: list[str] = []
    fbs: list[int] = []
    for u in urls:
        k, fb = leak_safe_group_key(u)
        keys.append(k)
        fbs.append(int(fb))
    return (
        pd.Series(keys, index=df.index),
        pd.Series(fbs, index=df.index, dtype=int),
    )


def ensure_group_column(df: pd.DataFrame) -> pd.Series:
    """Same keys used for StratifiedGroupKFold (for audits / leakage_report)."""
    g, _ = _leak_group_and_fallback(df)
    return g


def stratified_group_train_test(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Single train/test split; groups never appear in both sets."""
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    groups, fallback = _leak_group_and_fallback(df)
    df["_leak_group"] = groups

    y = df["label"].values
    n_splits = max(3, min(10, int(round(1.0 / max(test_size, 0.05)))))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(df)), y, df["_leak_group"].values))
    train = df.iloc[train_idx].drop(columns=["_leak_group"]).copy()
    test = df.iloc[test_idx].drop(columns=["_leak_group"]).copy()
    train["group_key_fallback_used"] = fallback.iloc[train_idx].values
    test["group_key_fallback_used"] = fallback.iloc[test_idx].values

    stats: Dict[str, Any] = {
        "split_method": "StratifiedGroupKFold_first_fold",
        "n_splits_param": n_splits,
        "approx_test_fraction": 1.0 / n_splits,
        "n_train": len(train),
        "n_test": len(test),
        "train_groups": int(ensure_group_column(train).nunique()),
        "test_groups": int(ensure_group_column(test).nunique()),
    }
    return train, test, stats


def split_leak_safe(
    enriched_csv: Optional[Path] = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    train_out: Optional[Path] = None,
    test_out: Optional[Path] = None,
) -> Tuple[Path, Path]:
    ensure_layout()
    path = enriched_csv or (processed_dir() / "enriched.csv")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if df.empty or "label" not in df.columns:
        tr = train_out or (processed_dir() / "train.csv")
        te = test_out or (processed_dir() / "test.csv")
        df.to_csv(tr, index=False)
        df.to_csv(te, index=False)
        return tr, te

    train, test, st = stratified_group_train_test(df, test_size=test_size, random_state=random_state)
    tr_path = train_out or (processed_dir() / "train.csv")
    te_path = test_out or (processed_dir() / "test.csv")
    train.to_csv(tr_path, index=False)
    test.to_csv(te_path, index=False)

    # overlap audit
    c_train = set(train["canonical_url"].astype(str))
    c_test = set(test["canonical_url"].astype(str))
    st["canonical_url_overlap_count"] = len(c_train & c_test)
    g_train = set(ensure_group_column(train))
    g_test = set(ensure_group_column(test))
    st["registered_domain_overlap_count"] = len(g_train & g_test)
    st["train_label_counts"] = train["label"].value_counts().sort_index().to_dict()
    st["test_label_counts"] = test["label"].value_counts().sort_index().to_dict()

    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "split_leak_safe_stats.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    logger.info(
        "Leak-safe split train=%s test=%s url_overlap=%s domain_overlap=%s",
        len(train),
        len(test),
        st["canonical_url_overlap_count"],
        st["registered_domain_overlap_count"],
    )
    return tr_path, te_path


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "split_leak_safe.log")
    ap = argparse.ArgumentParser(description="StratifiedGroupKFold split (primary Kaggle path).")
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-out", type=Path, default=None)
    ap.add_argument("--test-out", type=Path, default=None)
    args = ap.parse_args()
    split_leak_safe(
        args.input,
        test_size=args.test_size,
        random_state=args.seed,
        train_out=args.train_out,
        test_out=args.test_out,
    )


if __name__ == "__main__":
    main()
