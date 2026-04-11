"""Stratified train/test split with group holdout (registered_domain) to limit campaign leakage."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold

from src.pipeline.paths import ensure_layout, processed_dir, reports_dir

logger = logging.getLogger(__name__)


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _registered_domain(url: str) -> str:
    host = _host(url)
    if not host:
        return "missing_host"
    ext = tldextract.extract(host)
    reg = ".".join(p for p in (ext.domain, ext.suffix) if p)
    return reg.lower() or host


def ensure_group_column(df: pd.DataFrame) -> pd.Series:
    if "registered_domain" in df.columns and df["registered_domain"].notna().any():
        s = df["registered_domain"].astype(str).str.strip().str.lower()
        s = s.replace("", np.nan).fillna(df["canonical_url"].astype(str).map(_registered_domain))
        return s
    return df["canonical_url"].astype(str).map(_registered_domain)


def stratified_group_train_test(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Single train/test split; groups never appear in both sets."""
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    groups = ensure_group_column(df)
    df["_leak_group"] = groups

    y = df["label"].values
    n_splits = max(3, min(10, int(round(1.0 / max(test_size, 0.05)))))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(df)), y, df["_leak_group"].values))
    train = df.iloc[train_idx].drop(columns=["_leak_group"])
    test = df.iloc[test_idx].drop(columns=["_leak_group"])

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
