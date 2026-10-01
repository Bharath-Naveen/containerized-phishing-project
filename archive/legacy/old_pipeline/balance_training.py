"""Downsample phishing in the training split to a configurable multiple of legit rows."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.pipeline.paths import processed_dir, reports_dir


def _train_label_balance_dict(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "label" not in df.columns:
        return {"n_rows": 0, "legit_0": 0, "phish_1": 0, "ratio_phish_per_legit": None}
    lab = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())
    return {
        "n_rows": len(df),
        "legit_0": n0,
        "phish_1": n1,
        "ratio_phish_per_legit": round(n1 / max(n0, 1), 4) if n0 else None,
    }


def _patch_split_balance_report(final_train_df: pd.DataFrame) -> None:
    p = reports_dir() / "split_balance_report.json"
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    data["after_train_phish_downsample"] = {"train": _train_label_balance_dict(final_train_df)}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

logger = logging.getLogger(__name__)


def _group_key_series(df: pd.DataFrame) -> pd.Series:
    rd = df.get("registered_domain", pd.Series([""] * len(df))).astype(str).str.strip().str.lower()
    canon = df.get("canonical_url", pd.Series([""] * len(df))).astype(str)

    def one_row(i: int) -> str:
        r = rd.iloc[i]
        if r and r != "nan":
            return r
        c = canon.iloc[i]
        host = c.split("://")[-1].split("/")[0].lower() if c else "missing"
        return host or "missing"

    return pd.Series([one_row(i) for i in range(len(df))], index=df.index)


def downsample_train_phish(
    train_df: pd.DataFrame,
    *,
    phish_to_legit_ratio: float,
    random_state: int = 42,
    group_aware: bool = True,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Keep all legit (label 0); cap phishing at floor(phish_to_legit_ratio * n_legit).

    phish_to_legit_ratio: e.g. 2.0 => at most 2 phishing rows per legit row.
    If group_aware, fill the cap by iterating shuffled registered-domain groups
    (same key as split.py) and taking rows until the cap is reached.
    """
    stats: Dict[str, Any] = {
        "phish_to_legit_ratio_requested": phish_to_legit_ratio,
        "group_aware": group_aware,
    }
    df = train_df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    legit_mask = df["label"] == 0
    phish_mask = df["label"] == 1
    n_legit = int(legit_mask.sum())
    n_phish = int(phish_mask.sum())
    stats["train_rows_before"] = len(df)
    stats["legit_before"] = n_legit
    stats["phish_before"] = n_phish

    if n_legit == 0 or n_phish == 0 or phish_to_legit_ratio <= 0:
        stats["skipped"] = True
        stats["reason"] = "no_legit_or_no_phish_or_nonpositive_ratio"
        return df, stats

    max_phish = int(np.floor(float(phish_to_legit_ratio) * n_legit))
    max_phish = max(0, max_phish)
    stats["max_phish_after_balance"] = max_phish

    if n_phish <= max_phish:
        stats["skipped"] = True
        stats["reason"] = "already_under_cap"
        stats["phish_after"] = n_phish
        stats["train_rows_after"] = len(df)
        return df, stats

    rng = np.random.RandomState(random_state)
    phish_idx = df.index[phish_mask].tolist()
    keep_phish: Set[Any] = set()

    if group_aware:
        sub = df.loc[phish_mask]
        groups = _group_key_series(sub)
        by_g: DefaultDict[str, List[Any]] = defaultdict(list)
        for idx, g in zip(sub.index.tolist(), groups.tolist()):
            by_g[g].append(idx)
        g_list = list(by_g.keys())
        rng.shuffle(g_list)
        for g in g_list:
            ids = by_g[g]
            room = max_phish - len(keep_phish)
            if room <= 0:
                break
            if len(ids) <= room:
                keep_phish.update(ids)
            else:
                picked = rng.choice(ids, size=room, replace=False).tolist()
                keep_phish.update(picked)
                break
        if len(keep_phish) < max_phish:
            remaining = [i for i in phish_idx if i not in keep_phish]
            need = max_phish - len(keep_phish)
            if remaining and need > 0:
                extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist()
                keep_phish.update(extra)
    else:
        picked = rng.choice(phish_idx, size=max_phish, replace=False).tolist()
        keep_phish = set(picked)

    keep_mask = legit_mask | df.index.isin(list(keep_phish))
    out = df.loc[keep_mask].reset_index(drop=True)
    stats["skipped"] = False
    stats["phish_dropped"] = n_phish - len(keep_phish)
    stats["phish_after"] = int((out["label"] == 1).sum())
    stats["legit_after"] = int((out["label"] == 0).sum())
    stats["train_rows_after"] = len(out)
    if stats["legit_after"]:
        stats["ratio_phish_per_legit"] = round(stats["phish_after"] / stats["legit_after"], 4)
    else:
        stats["ratio_phish_per_legit"] = None
    return out, stats


def apply_train_balance_to_csv(
    train_csv: Optional[Path] = None,
    *,
    phish_to_legit_ratio: Optional[float],
    random_state: int = 42,
    group_aware: bool = True,
) -> Dict[str, Any]:
    """Read train.csv, optionally downsample phishing, write back. Returns stats for reporting."""
    path = train_csv or (processed_dir() / "train.csv")
    if phish_to_legit_ratio is None:
        summary = {
            "train_balance_applied": False,
            "reason": "phish_to_legit_ratio not set",
            "train_csv": str(path),
        }
        (reports_dir() / "train_balance_stats.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    df = pd.read_csv(path, dtype=str, low_memory=False)
    if df.empty:
        summary = {"train_balance_applied": False, "error": "empty_train_csv", "train_csv": str(path)}
        (reports_dir() / "train_balance_stats.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    out, st = downsample_train_phish(
        df,
        phish_to_legit_ratio=phish_to_legit_ratio,
        random_state=random_state,
        group_aware=group_aware,
    )
    out.to_csv(path, index=False)
    summary = {"train_balance_applied": True, "train_csv": str(path), **st}
    (reports_dir() / "train_balance_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _patch_split_balance_report(out)
    logger.info(
        "Train balance: legit=%s phish=%s -> phish=%s (ratio=%s)",
        st.get("legit_before"),
        st.get("phish_before"),
        st.get("phish_after"),
        phish_to_legit_ratio,
    )
    return summary
