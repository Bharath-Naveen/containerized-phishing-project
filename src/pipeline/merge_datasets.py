"""Merge Kaggle and fresh datasets with leakage-safe filtering."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.pipeline.fresh_data import deduplicate_urls, get_registered_domain, label_sanity_check

logger = logging.getLogger(__name__)


def merge_datasets(
    kaggle_df: pd.DataFrame,
    fresh_df: pd.DataFrame,
    *,
    fresh_holdout_df: Optional[pd.DataFrame] = None,
    fresh_weight: float = 0.3,
    random_state: int = 42,
    return_stats: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    kag = kaggle_df.copy()
    fr = fresh_df.copy()
    if "registered_domain" not in kag.columns:
        kag["registered_domain"] = kag["url"].map(get_registered_domain)
    if "registered_domain" not in fr.columns:
        fr["registered_domain"] = fr["url"].map(get_registered_domain)
    kag["registered_domain"] = kag["registered_domain"].fillna("").astype(str).str.lower()
    fr["registered_domain"] = fr["registered_domain"].fillna("").astype(str).str.lower()

    hold_domains = set()
    if fresh_holdout_df is not None and not fresh_holdout_df.empty:
        hd = fresh_holdout_df.copy()
        if "registered_domain" not in hd.columns:
            hd["registered_domain"] = hd["url"].map(get_registered_domain)
        hold_domains = set(hd["registered_domain"].fillna("").astype(str).str.lower().tolist())
        hold_domains.discard("")

    kag_domains = set(kag["registered_domain"].tolist())
    fresh_rows_available = int(len(fr))
    before_kag_overlap_filter = int(len(fr))
    fr = fr[~fr["registered_domain"].isin(kag_domains)].copy()
    fresh_vs_kaggle_domain_overlap_removed = before_kag_overlap_filter - int(len(fr))
    if hold_domains:
        before_holdout_overlap_filter = int(len(fr))
        fr = fr[~fr["registered_domain"].isin(hold_domains)].copy()
        fresh_vs_holdout_domain_overlap_removed = before_holdout_overlap_filter - int(len(fr))
    else:
        fresh_vs_holdout_domain_overlap_removed = 0
    fresh_rows_after_overlap = int(len(fr))

    cap = max(0, int(len(kag) * max(0.0, float(fresh_weight))))
    fresh_rows_used = 0
    if cap == 0:
        merged = kag.copy()
    else:
        fr = fr.head(min(cap, len(fr))).copy()
        fresh_rows_used = int(len(fr))
        merged = pd.concat([kag, fr], ignore_index=True)

    merged = deduplicate_urls(merged)
    merged = label_sanity_check(merged)
    merged = merged.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    stats: Dict[str, Any] = {
        "kaggle_rows": int(len(kag)),
        "fresh_rows_available": fresh_rows_available,
        "fresh_rows_after_overlap_filter": fresh_rows_after_overlap,
        "fresh_rows_used": fresh_rows_used,
        "combined_rows": int(len(merged)),
        "fresh_vs_kaggle_domain_overlap_removed": int(fresh_vs_kaggle_domain_overlap_removed),
        "fresh_vs_holdout_domain_overlap_removed": int(fresh_vs_holdout_domain_overlap_removed),
    }
    logger.info(
        "Merged dataset kaggle_rows=%s fresh_rows_available=%s fresh_rows_used=%s combined_rows=%s",
        stats["kaggle_rows"],
        stats["fresh_rows_available"],
        stats["fresh_rows_used"],
        stats["combined_rows"],
    )
    if return_stats:
        return merged, stats
    return merged
