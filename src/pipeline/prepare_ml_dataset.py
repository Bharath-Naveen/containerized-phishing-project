"""Filter enriched rows for more realistic ML training (reduce fetch-artifact + obvious lexical phish)."""

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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def prepare_ml_dataset(
    enriched_csv: Optional[Path] = None,
    *,
    max_phish_fetch_fail_fraction: float = 0.28,
    drop_obvious_lexical_phish: bool = True,
    random_state: int = 42,
) -> Tuple[Path, Dict[str, Any]]:
    """
    - Cap fraction of phishing rows with failed page fetch (reduces http_error dominating class 1).
    - Optionally drop obvious URL tricks (IP host, @-in-URL) for phishing only.
    - Keeps all legit rows (label 0).
    """
    ensure_layout()
    path = enriched_csv or (processed_dir() / "enriched.csv")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    stats: Dict[str, Any] = {"input": str(path), "input_rows": len(df)}

    if df.empty or "label" not in df.columns:
        out_path = processed_dir() / "enriched_ml_ready.csv"
        df.to_csv(out_path, index=False)
        stats["output_rows"] = 0
        stats["balance_after_prepare_ml"] = {
            "legit_0": 0,
            "phish_1": 0,
            "phish_share": 0.0,
            "ratio_phish_per_legit": None,
        }
        (reports_dir() / "ml_prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return out_path, stats

    df = df.copy()
    df["label"] = _to_num(df["label"]).fillna(1).astype(int)

    if drop_obvious_lexical_phish:
        ph = df["label"] == 1
        obvious = pd.Series(False, index=df.index)
        if "has_ip_address" in df.columns:
            obvious |= ph & (_to_num(df["has_ip_address"]).fillna(0) == 1)
        if "has_at_symbol" in df.columns:
            obvious |= ph & (_to_num(df["has_at_symbol"]).fillna(0) == 1)
        stats["dropped_obvious_lexical_phish"] = int(obvious.sum())
        df = df.loc[~obvious].copy()
    else:
        stats["dropped_obvious_lexical_phish"] = 0

    df_f = df.reset_index(drop=True)
    phish = df_f["label"] == 1
    stats["phish_fetch_fail_before"] = 0
    stats["phish_fetch_ok_before"] = 0
    dropped_fail = 0

    if phish.any() and "page_fetch_success" in df_f.columns:
        pfs = _to_num(df_f.loc[phish, "page_fetch_success"]).fillna(0).astype(int)
        fail_mask = pfs == 0
        n_fail = int(fail_mask.sum())
        n_ok = int((~fail_mask).sum())
        stats["phish_fetch_fail_before"] = n_fail
        stats["phish_fetch_ok_before"] = n_ok
        max_fail_allowed = int(np.ceil(max_phish_fetch_fail_fraction * phish.sum()))
        if n_fail > max_fail_allowed and n_fail > 0:
            rng = np.random.default_rng(random_state)
            phish_positions = np.where(phish.values)[0]
            fail_local = np.where(fail_mask.values)[0]
            fail_rows = phish_positions[fail_local]
            drop_n = n_fail - max_fail_allowed
            drop_rows = rng.choice(fail_rows, size=drop_n, replace=False)
            df_f = df_f.drop(index=drop_rows).reset_index(drop=True)
            dropped_fail = int(drop_n)
        stats["dropped_phish_fetch_failed_cap"] = dropped_fail
        phish = df_f["label"] == 1
        if phish.any():
            pfs2 = _to_num(df_f.loc[phish, "page_fetch_success"]).fillna(0).astype(int)
            stats["phish_fetch_fail_after"] = int((pfs2 == 0).sum())
            stats["phish_fetch_ok_after"] = int((pfs2 == 1).sum())
    else:
        stats["dropped_phish_fetch_failed_cap"] = 0

    out_path = processed_dir() / "enriched_ml_ready.csv"
    df_f.to_csv(out_path, index=False)
    stats["output_rows"] = len(df_f)
    stats["output_legit"] = int((df_f["label"] == 0).sum())
    stats["output_phish"] = int((df_f["label"] == 1).sum())
    ol, op = stats["output_legit"], stats["output_phish"]
    stats["balance_after_prepare_ml"] = {
        "legit_0": ol,
        "phish_1": op,
        "phish_share": round(op / max(len(df_f), 1), 6) if len(df_f) else 0.0,
        "ratio_phish_per_legit": round(op / max(ol, 1), 4) if ol else None,
    }
    (reports_dir() / "ml_prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("ML-ready rows=%s (legit=%s phish=%s)", len(df_f), stats["output_legit"], stats["output_phish"])
    return out_path, stats


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "prepare_ml_dataset.log")
    p = argparse.ArgumentParser(description="Build enriched_ml_ready.csv for training.")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--max-phish-fetch-fail-frac", type=float, default=0.28)
    p.add_argument("--keep-obvious-lexical", action="store_true", help="Do not drop IP/@ phishing URLs")
    args = p.parse_args()
    prepare_ml_dataset(
        args.input,
        max_phish_fetch_fail_fraction=args.max_phish_fetch_fail_frac,
        drop_obvious_lexical_phish=not args.keep_obvious_lexical,
    )


if __name__ == "__main__":
    main()
