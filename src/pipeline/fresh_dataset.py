"""Optional integration of externally collected phishing dataset into Kaggle-normalized training rows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import tldextract

from src.pipeline.label_policy import kaggle_status_to_internal
from src.pipeline.paths import data_dir, outputs_dir, project_root

logger = logging.getLogger(__name__)

PHISH_SOURCES = {"phishstats", "phishtank", "openphish", "phreshphish"}
LEGIT_SOURCES = {"tranco", "brand_official", "curated_legit", "curated_legitimate"}


def _reg_domain(url: str) -> str:
    ext = tldextract.extract(str(url or ""))
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return (ext.domain or "").lower()


def _status_from_fresh_row(row: pd.Series) -> Optional[int]:
    src = str(row.get("source") or "").strip().lower()
    raw_label = row.get("label")
    if src in PHISH_SOURCES:
        return 0  # Kaggle convention: phishing=0
    if src in LEGIT_SOURCES:
        return 1  # Kaggle convention: legitimate=1
    try:
        lbl = int(raw_label)
    except Exception:
        return None
    if lbl == 1:
        return 0
    if lbl == 0:
        return 1
    return None


def _counts_dict(s: pd.Series) -> Dict[str, int]:
    vc = s.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def _series_or_empty(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([""] * len(df), index=df.index)


def load_and_merge_fresh_dataset(
    *,
    kaggle_normalized_csv: Path,
    fresh_dataset_csv: Path,
    fresh_recent_holdout_csv: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Merge Kaggle-normalized rows with optional fresh dataset.

    Output keeps project-internal `label` convention unchanged (0 legit / 1 phish)
    while preserving an explicit Kaggle-compatible `status` column (0 phish / 1 legit).
    """
    kag = pd.read_csv(kaggle_normalized_csv, dtype=str, low_memory=False)
    kag["url"] = kag["url"].astype(str).str.strip()
    kag["label"] = pd.to_numeric(kag["label"], errors="coerce").fillna(0).astype(int)
    kag["status"] = kag["label"].map(lambda x: 1 if int(x) == 0 else 0)

    fresh = pd.read_csv(fresh_dataset_csv, dtype=str, low_memory=False)
    if "url" not in fresh.columns:
        raise ValueError(f"Fresh dataset missing required `url` column: {fresh_dataset_csv}")
    fresh["url"] = fresh["url"].astype(str).str.strip()
    fresh = fresh[fresh["url"].str.len() > 0].copy()

    holdout_df = pd.DataFrame()
    if fresh_recent_holdout_csv is not None and fresh_recent_holdout_csv.is_file():
        holdout_df = pd.read_csv(fresh_recent_holdout_csv, dtype=str, low_memory=False)
        if "url" in holdout_df.columns:
            holdout_df["url"] = holdout_df["url"].astype(str).str.strip()
            holdout_urls = set(holdout_df["url"].tolist())
            if holdout_urls:
                fresh = fresh[~fresh["url"].isin(holdout_urls)].copy()
        if "registered_domain" in holdout_df.columns:
            holdout_domains = set(
                holdout_df["registered_domain"].fillna("").astype(str).str.strip().str.lower().tolist()
            )
            holdout_domains.discard("")
            if holdout_domains:
                fresh_domains = fresh.get("registered_domain", pd.Series([""] * len(fresh), index=fresh.index))
                fresh_domains = fresh_domains.fillna("").astype(str).str.strip().str.lower()
                fresh = fresh[~fresh_domains.isin(holdout_domains)].copy()

    fresh["source"] = _series_or_empty(fresh, "source").astype(str).str.strip()
    fresh["brand_target"] = _series_or_empty(fresh, "brand_target").astype(str).str.strip()
    fresh["collection_date"] = _series_or_empty(fresh, "collection_date").astype(str).str.strip()
    fresh["notes"] = _series_or_empty(fresh, "notes").astype(str).str.strip()
    fresh["registered_domain"] = _series_or_empty(fresh, "registered_domain").astype(str).str.strip().str.lower()
    needs_reg = fresh["registered_domain"].eq("") | fresh["registered_domain"].isna()
    if bool(needs_reg.any()):
        fresh.loc[needs_reg, "registered_domain"] = fresh.loc[needs_reg, "url"].map(_reg_domain)

    fresh["status"] = fresh.apply(_status_from_fresh_row, axis=1)
    fresh = fresh[fresh["status"].notna()].copy()
    fresh["status"] = fresh["status"].astype(int)
    bad_status = sorted(set(fresh["status"].tolist()) - {0, 1})
    if bad_status:
        raise ValueError(f"Fresh dataset produced invalid status values: {bad_status}")
    fresh["label"] = fresh["status"].map(kaggle_status_to_internal).astype(int)

    # source consistency checks
    source_l = fresh["source"].str.lower()
    must_phish = fresh[source_l.isin(PHISH_SOURCES)]
    must_legit = fresh[source_l.isin(LEGIT_SOURCES)]
    if not must_phish.empty and not bool((must_phish["status"] == 0).all()):
        raise ValueError("Fresh phishing-source rows violated status=0 mapping.")
    if not must_legit.empty and not bool((must_legit["status"] == 1).all()):
        raise ValueError("Fresh legit-source rows violated status=1 mapping.")

    fresh_norm = pd.DataFrame(
        {
            "url": fresh["url"],
            "label": fresh["label"],
            "status": fresh["status"],
            "source_file": Path(fresh_dataset_csv).name,
            "source_dataset": "fresh_phishstats_extension",
            "source_brand_hint": fresh["brand_target"],
            "action_category": "fresh_extension",
            "kaggle_raw_status": fresh["status"],
            "fresh_source": fresh["source"],
            "fresh_brand_target": fresh["brand_target"],
            "fresh_collection_date": fresh["collection_date"],
            "fresh_registered_domain": fresh["registered_domain"],
            "fresh_notes": fresh["notes"],
        }
    )

    combined = pd.concat([kag, fresh_norm], ignore_index=True)
    combined["url"] = combined["url"].astype(str).str.strip()
    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    dedup_removed = before - len(combined)

    combined_dir = data_dir() / "combined_dataset"
    combined_dir.mkdir(parents=True, exist_ok=True)
    out_csv = combined_dir / "kaggle_plus_fresh_normalized.csv"
    combined.to_csv(out_csv, index=False)

    if not holdout_df.empty:
        holdout_out = combined_dir / "fresh_recent_holdout.csv"
        holdout_df.to_csv(holdout_out, index=False)

    sanity: Dict[str, Any] = {
        "kaggle_rows_in": int(len(kag)),
        "fresh_rows_in_after_holdout_exclusion": int(len(fresh_norm)),
        "combined_rows_out": int(len(combined)),
        "dedup_removed_exact_url": int(dedup_removed),
        "status_counts_combined": _counts_dict(combined["status"] if "status" in combined.columns else pd.Series([], dtype=object)),
        "label_counts_combined_internal": _counts_dict(combined["label"]),
        "fresh_status_by_source": (
            fresh.groupby(["source", "status"]).size().rename("n").reset_index().to_dict(orient="records")
            if not fresh.empty
            else []
        ),
        "fresh_status_by_brand_target": (
            fresh.groupby(["brand_target", "status"]).size().rename("n").reset_index().to_dict(orient="records")
            if not fresh.empty
            else []
        ),
    }
    rep_dir = outputs_dir() / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "fresh_dataset_sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    logger.info("Fresh+Kaggle merge complete -> %s", out_csv)
    logger.info("Combined status counts (Kaggle convention: 0=phish,1=legit): %s", sanity["status_counts_combined"])
    if not fresh.empty:
        logger.info(
            "Fresh rows by source/status: %s",
            fresh.groupby(["source", "status"]).size().rename("n").reset_index().to_dict(orient="records"),
        )
        logger.info(
            "Fresh rows by brand_target/status: %s",
            fresh.groupby(["brand_target", "status"]).size().rename("n").reset_index().to_dict(orient="records"),
        )
    return out_csv, sanity


def default_fresh_dataset_path() -> Path:
    return project_root() / "phishing_dataset" / "dataset_full.csv"


def default_fresh_recent_holdout_path() -> Path:
    return project_root() / "phishing_dataset" / "dataset_test_recent.csv"
