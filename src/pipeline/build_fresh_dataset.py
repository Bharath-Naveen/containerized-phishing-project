"""Build fresh phishing/legit dataset for augmentation and holdout."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd

from src.pipeline.fresh_data import (
    PHISH_STATUS,
    LEGIT_STATUS,
    cap_per_domain,
    collect_phishstats,
    collect_tranco,
    deduplicate_urls,
    label_sanity_check,
)

logger = logging.getLogger(__name__)


def _balance_no_upsample(df: pd.DataFrame) -> pd.DataFrame:
    ph = df[df["status"] == PHISH_STATUS].copy()
    lg = df[df["status"] == LEGIT_STATUS].copy()
    if ph.empty or lg.empty:
        return df.copy()
    n = min(len(ph), len(lg))
    ph = ph.head(n)
    lg = lg.head(n)
    out = pd.concat([ph, lg], ignore_index=True)
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_fresh_dataset(
    *,
    n_phishing: int = 2000,
    n_legitimate: int = 2000,
    phish_pages: int = 12,
    phish_timeout_s: int = 20,
    phish_max_failed_pages: int = 3,
    tranco_local_csv_path: Optional[Path] = None,
    max_per_domain: int = 10,
    holdout_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ph, ph_meta = collect_phishstats(
        pages=phish_pages,
        timeout_s=phish_timeout_s,
        max_rows=n_phishing,
        max_failed_pages=phish_max_failed_pages,
        return_meta=True,
    )
    lg, tr_meta = collect_tranco(
        n=n_legitimate,
        local_csv_path=tranco_local_csv_path,
        return_meta=True,
    )
    merged = pd.concat([ph, lg], ignore_index=True)
    merged = deduplicate_urls(merged)
    merged = cap_per_domain(merged, max_per_domain=max_per_domain)
    merged = label_sanity_check(merged)
    merged = _balance_no_upsample(merged)
    collection_meta: dict[str, Any] = {
        "phishstats_rows_collected": int(ph_meta.get("phishstats_rows_collected", len(ph))),
        "phishstats_errors_count": int(ph_meta.get("phishstats_fetch_errors", 0)),
        "tranco_rows_collected": int(tr_meta.get("tranco_rows_collected", len(lg))),
        "tranco_download_failed": bool(tr_meta.get("tranco_download_failed", False)),
        "tranco_error": tr_meta.get("tranco_error"),
    }
    if merged.empty:
        return merged.copy(), merged.copy(), collection_meta

    merged["collection_date"] = merged["collection_date"].fillna("").astype(str)
    parsed = pd.to_datetime(merged["collection_date"], errors="coerce")
    merged["_cd"] = parsed.fillna(pd.Timestamp(datetime.utcnow()))
    merged = merged.sort_values("_cd").reset_index(drop=True)

    cut = max(1, int(len(merged) * max(0.0, min(0.9, holdout_frac))))
    holdout = merged.tail(cut).drop(columns=["_cd"]).reset_index(drop=True)
    train = merged.head(len(merged) - cut).drop(columns=["_cd"]).reset_index(drop=True)
    logger.info(
        "Fresh dataset built train=%s holdout=%s status_counts=%s",
        len(train),
        len(holdout),
        train["status"].value_counts().to_dict() if not train.empty else {},
    )
    return train, holdout, collection_meta
