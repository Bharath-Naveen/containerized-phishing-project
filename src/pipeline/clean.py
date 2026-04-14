"""URL cleaning, validation, deduplication."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.pipeline.paths import ensure_layout, interim_dir, processed_dir
from src.pipeline.safe_url import canonicalize_url_safe

logger = logging.getLogger(__name__)


def canonicalize_url(raw: str) -> Tuple[str, int, str]:
    """Return (canonical_url, invalid_url_flag, parse_error)."""
    return canonicalize_url_safe(raw)


def clean(
    normalized_csv: Optional[Path] = None,
    *,
    output_csv: Optional[Path] = None,
    stats_json: Optional[Path] = None,
) -> pd.DataFrame:
    ensure_layout()
    path = normalized_csv or (interim_dir() / "normalized.csv")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if "action_category" not in df.columns:
        df["action_category"] = "uncategorized"
    if df.empty:
        df = df.copy()
        df["canonical_url"] = []
        df["invalid_url"] = []
        df["parse_error"] = []
        dedup_stats = {"canonical_url_rows_before": 0, "canonical_url_rows_after": 0}
    else:
        cans = []
        inv = []
        errs = []
        for u in df["url"].fillna(""):
            c, i, e = canonicalize_url(str(u))
            cans.append(c)
            inv.append(i)
            errs.append(e)
        df = df.copy()
        df["canonical_url"] = cans
        df["invalid_url"] = inv
        df["parse_error"] = errs
        before = len(df)
        df = df.drop_duplicates(subset=["canonical_url"], keep="first")
        after = len(df)
        logger.info("Deduplicated canonical_url: %s -> %s rows", before, after)
        dedup_stats = {"canonical_url_rows_before": before, "canonical_url_rows_after": after}

    out_path = output_csv or (processed_dir() / "cleaned.csv")
    processed_dir().mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote cleaned dataset -> %s", out_path)
    if stats_json:
        import json

        payload = {"input": str(path), "output": str(out_path), **dedup_stats}
        stats_json.parent.mkdir(parents=True, exist_ok=True)
        stats_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return df


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "clean.log")
    p = argparse.ArgumentParser(description="Clean and dedupe URLs.")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--stats-json", type=Path, default=None)
    args = p.parse_args()
    clean(args.input, output_csv=args.output, stats_json=args.stats_json)


if __name__ == "__main__":
    main()
