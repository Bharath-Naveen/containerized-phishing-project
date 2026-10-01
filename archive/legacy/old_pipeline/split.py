"""Grouped splits with per-label group holdout for approximate class balance."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.paths import ensure_layout, processed_dir, reports_dir

logger = logging.getLogger(__name__)


def _split_label_balance(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "label" not in df.columns:
        return {"legit_0": 0, "phish_1": 0, "n_rows": 0, "ratio_phish_per_legit": None}
    lab = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())
    return {
        "n_rows": len(df),
        "legit_0": n0,
        "phish_1": n1,
        "ratio_phish_per_legit": round(n1 / max(n0, 1), 4) if n0 else None,
    }


def default_enriched_path() -> Path:
    """Prefer ``enriched_ml_ready.csv`` when present and non-empty."""
    ml_ready = processed_dir() / "enriched_ml_ready.csv"
    if ml_ready.exists():
        try:
            head = pd.read_csv(ml_ready, nrows=2, dtype=str)
            if len(head) > 0:
                return ml_ready
        except Exception:
            pass
    return processed_dir() / "enriched.csv"


def _group_key(row: pd.Series) -> str:
    rd = str(row.get("registered_domain") or "").strip().lower()
    if rd and rd != "nan":
        return rd
    canon = str(row.get("canonical_url") or "")
    host = canon.split("://")[-1].split("/")[0].lower() if canon else "missing"
    return host or "missing"


def _balanced_group_train_test(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each label, assign ~test_size fraction of *groups* to test (by shuffled group list).
    Same registered_domain never appears in both train and test. Aims for similar
    train/test phishing:legit *group* fractions when both classes have enough groups.
    """
    rng = np.random.RandomState(random_state)
    test_groups: set = set()
    labels = sorted(df["label"].unique())
    for lab in labels:
        sub = df[df["label"] == lab]
        groups = sub["_group"].unique()
        groups = list(groups)
        rng.shuffle(groups)
        if len(groups) <= 1:
            continue
        n_test = int(round(len(groups) * test_size))
        n_test = max(1, min(n_test, len(groups) - 1))
        test_groups.update(groups[:n_test])

    te_mask = df["_group"].isin(test_groups)
    train = df.loc[~te_mask].drop(columns=["_group"])
    test = df.loc[te_mask].drop(columns=["_group"])
    if len(train) == 0 or len(test) == 0:
        from sklearn.model_selection import train_test_split

        strat = df["label"] if df["label"].nunique() > 1 else None
        tr, te = train_test_split(
            df.drop(columns=["_group"]),
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
        return tr, te
    return train, test


def split_dataset(
    enriched_csv: Optional[Path] = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    brand_holdout: Optional[str] = None,
    balanced_per_label_groups: bool = True,
    train_phish_legit_ratio: Optional[float] = None,
    train_balance_random_state: int = 42,
    train_balance_group_aware: bool = True,
) -> tuple[Path, Path]:
    ensure_layout()
    path = enriched_csv or default_enriched_path()
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if df.empty:
        tr = processed_dir() / "train.csv"
        te = processed_dir() / "test.csv"
        df.to_csv(tr, index=False)
        df.to_csv(te, index=False)
        return tr, te

    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    df["_group"] = df.apply(_group_key, axis=1)

    if brand_holdout:
        bh = brand_holdout.lower()
        mask = df["source_brand_hint"].astype(str).str.lower() == bh
        test = df[mask].copy()
        train = df[~mask].copy()
        train = train.drop(columns=["_group"], errors="ignore")
        test = test.drop(columns=["_group"], errors="ignore")
    elif balanced_per_label_groups and len(df) >= 10 and df["label"].nunique() > 1:
        train, test = _balanced_group_train_test(df, test_size=test_size, random_state=random_state)
    else:
        use_grouped = df["_group"].nunique() > 1
        if use_grouped:
            try:
                from sklearn.model_selection import StratifiedGroupKFold

                n_splits = min(5, max(2, int(round(1 / max(test_size, 0.05)))))
                sgkf = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state
                )
                idx = np.arange(len(df))
                split_iter = sgkf.split(idx, df["label"].values, df["_group"].values)
                train_idx, test_idx = next(split_iter)
                train = df.iloc[train_idx].drop(columns=["_group"])
                test = df.iloc[test_idx].drop(columns=["_group"])
            except Exception as e:
                logger.warning("StratifiedGroupKFold failed (%s); using random split", e)
                use_grouped = False
        if not use_grouped:
            from sklearn.model_selection import train_test_split

            strat = df["label"] if df["label"].nunique() > 1 else None
            train, test = train_test_split(
                df.drop(columns=["_group"]),
                test_size=test_size,
                random_state=random_state,
                stratify=strat,
            )

    tr_path = processed_dir() / "train.csv"
    te_path = processed_dir() / "test.csv"
    train.to_csv(tr_path, index=False)
    test.to_csv(te_path, index=False)
    n0_tr = int((train["label"].astype(int) == 0).sum())
    n1_tr = int((train["label"].astype(int) == 1).sum())
    n0_te = int((test["label"].astype(int) == 0).sum())
    n1_te = int((test["label"].astype(int) == 1).sum())
    logger.info(
        "Train rows=%s (legit=%s phish=%s) | Test rows=%s (legit=%s phish=%s)",
        len(train),
        n0_tr,
        n1_tr,
        len(test),
        n0_te,
        n1_te,
    )
    split_report = {
        "input_csv": str(path),
        "after_split_before_train_balance": {
            "train": _split_label_balance(train),
            "test": _split_label_balance(test),
        },
    }
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "split_balance_report.json").write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )

    from src.pipeline.balance_training import apply_train_balance_to_csv

    apply_train_balance_to_csv(
        tr_path,
        phish_to_legit_ratio=train_phish_legit_ratio,
        random_state=train_balance_random_state,
        group_aware=train_balance_group_aware,
    )
    return tr_path, te_path


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "split.log")
    p = argparse.ArgumentParser(description="Create grouped train/test split.")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--brand-holdout", type=str, default=None)
    p.add_argument(
        "--no-balanced-label-groups",
        action="store_true",
        help="Use StratifiedGroupKFold on full table instead of per-label group fractions.",
    )
    p.add_argument(
        "--train-phish-legit-ratio",
        type=float,
        default=None,
        help="Cap training phishing rows at this multiple of training legit rows (e.g. 2.0 ≈ 2:1 phish:legit). "
        "Test split unchanged. Uses registered_domain-aware groups when possible.",
    )
    p.add_argument(
        "--train-balance-rowwise",
        action="store_true",
        help="Downsample phishing rows uniformly instead of group-aware packing.",
    )
    args = p.parse_args()
    split_dataset(
        args.input,
        test_size=args.test_size,
        brand_holdout=args.brand_holdout,
        balanced_per_label_groups=not args.no_balanced_label_groups,
        train_phish_legit_ratio=args.train_phish_legit_ratio,
        train_balance_group_aware=not args.train_balance_rowwise,
    )


if __name__ == "__main__":
    main()
