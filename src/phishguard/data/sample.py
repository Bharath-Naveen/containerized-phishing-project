"""Stratified row sampling (preserve approximate class balance) for large URL datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def stratified_sample_by_label(
    df: pd.DataFrame,
    *,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: int = 42,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Draw a stratified subset (same procedure as StratifiedShuffleSplit: per-class proportions
    match the full frame, size exactly ``n`` when possible).

    Specify exactly one of ``n`` or ``frac``.
    """
    if n is not None and frac is not None:
        raise ValueError("Provide only one of n or frac, not both")
    if n is None and frac is None:
        raise ValueError("Provide n or frac")
    if df.empty:
        return df.copy(), {"error": "empty_input", "sample_rows": 0}

    work = df.copy()
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce").fillna(1).astype(int)
    total = len(work)
    if frac is not None:
        target_n = max(1, int(round(total * float(frac))))
    else:
        target_n = int(n)  # type: ignore[arg-type]
    n = target_n
    n = min(n, total)
    if n == total:
        stats: Dict[str, Any] = {
            "mode": "no_sampling_needed",
            "input_rows": total,
            "sample_rows": n,
            "random_state": random_state,
            "label_counts_input": work[label_col].value_counts().sort_index().to_dict(),
            "label_counts_sample": work[label_col].value_counts().sort_index().to_dict(),
        }
        return work.reset_index(drop=True), stats

    y = work[label_col].values
    X = np.zeros((total, 1))

    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=random_state)
        train_idx, _ = next(sss.split(X, y))
    except ValueError as e:
        logger.warning("StratifiedShuffleSplit failed (%s); using simple random sample.", e)
        train_idx = work.sample(n=n, random_state=random_state).index.to_numpy()
        out = work.loc[train_idx].reset_index(drop=True)
        stats = {
            "mode": "random_fallback",
            "input_rows": total,
            "sample_rows": len(out),
            "random_state": random_state,
            "label_counts_input": work[label_col].value_counts().sort_index().to_dict(),
            "label_counts_sample": out[label_col].value_counts().sort_index().to_dict(),
            "fallback_reason": str(e),
        }
        return out, stats

    out = work.iloc[train_idx].reset_index(drop=True)
    stats = {
        "mode": "stratified",
        "input_rows": total,
        "sample_rows": len(out),
        "random_state": random_state,
        "label_counts_input": work[label_col].value_counts().sort_index().to_dict(),
        "label_counts_sample": out[label_col].value_counts().sort_index().to_dict(),
    }
    return out, stats


def save_sampled_cleaned_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    manifest: Dict[str, Any],
    manifest_path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["sampled_cleaned_csv"] = str(path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote stratified sample -> %s (rows=%s)", path, len(df))
    logger.info("Sample manifest -> %s", manifest_path)
    return path
