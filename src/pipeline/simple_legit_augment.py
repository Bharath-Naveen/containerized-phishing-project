"""Prepend curated legitimate URLs (large simple-legit set + hard-legit logins) to cleaned Kaggle frames."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.clean import canonicalize_url
from src.pipeline.label_policy import INTERNAL_LEGIT
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMPLE_LEGIT_JSONL = _REPO_ROOT / "data" / "evaluation" / "simple_legit_urls.jsonl"
DEFAULT_HARD_LEGIT_JSONL = _REPO_ROOT / "data" / "evaluation" / "hard_legit_urls.jsonl"


def _load_jsonl_url_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_simple_legit_rows(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or DEFAULT_SIMPLE_LEGIT_JSONL
    if not p.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def curated_legit_augment_rows(
    *,
    simple_jsonl: Optional[Path] = None,
    include_hard_legit: bool = True,
) -> List[Dict[str, Any]]:
    """Ordered list: simple-legit URLs then hard-legit (login/SaaS) URLs, de-duped by URL string."""
    seen: set[str] = set()
    combined: List[Dict[str, Any]] = []
    for rec in load_simple_legit_rows(simple_jsonl):
        u = str(rec.get("url") or "").strip()
        if u in seen:
            continue
        seen.add(u)
        combined.append(rec)
    if include_hard_legit:
        for rec in _load_jsonl_url_rows(DEFAULT_HARD_LEGIT_JSONL):
            u = str(rec.get("url") or "").strip()
            if u in seen:
                continue
            seen.add(u)
            combined.append(rec)
    return combined


def augment_cleaned_with_simple_legit(
    df: pd.DataFrame,
    *,
    jsonl_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepend label=0 rows for curated homepages / clean URLs not already present.

    Rows are **prepended** so they are enriched first (any ``--limit`` on enrich still sees them).

    Expects columns at least ``url``; adds ``canonical_url`` / ``invalid_url`` / ``parse_error`` if missing.
    """
    if df.empty or "url" not in df.columns:
        return df, {"skipped": True, "reason": "empty_or_no_url_column"}

    base = df.copy()
    if "canonical_url" not in base.columns:
        cans, invs, errs = [], [], []
        for u in base["url"].fillna("").astype(str):
            c, i, e = canonicalize_url(str(u))
            cans.append(c)
            invs.append(i)
            errs.append(e)
        base["canonical_url"] = cans
        base["invalid_url"] = invs
        base["parse_error"] = errs

    existing = set(base["canonical_url"].astype(str).str.strip())
    rows_in = curated_legit_augment_rows(simple_jsonl=jsonl_path, include_hard_legit=True)
    added = 0
    mini: List[Dict[str, Any]] = []
    col_list = list(base.columns)
    label_as_str = pd.api.types.is_string_dtype(base["label"]) or str(base["label"].dtype) == "object"

    for rec in rows_in:
        raw_u = str(rec.get("url") or "").strip()
        if not raw_u:
            continue
        c, inv, err = canonicalize_url(raw_u)
        if not c or str(c).strip() in existing:
            continue
        row = {col: "" for col in col_list}
        row["url"] = raw_u
        row["label"] = str(INTERNAL_LEGIT) if label_as_str else INTERNAL_LEGIT
        row["canonical_url"] = c
        row["invalid_url"] = inv
        row["parse_error"] = err
        if "source_file" in row:
            row["source_file"] = "simple_legit_urls.jsonl"
        if "source_dataset" in row:
            row["source_dataset"] = "curated_simple_legit"
        if "source_brand_hint" in row:
            row["source_brand_hint"] = ""
        if "action_category" in row:
            row["action_category"] = "curated_homepage_clean"
        if "kaggle_raw_status" in row:
            row["kaggle_raw_status"] = "1"
        mini.append(row)
        existing.add(str(c).strip())
        added += 1

    if not mini:
        return base, {"rows_added": 0, "jsonl_rows_considered": len(rows_in)}

    aug = pd.concat([pd.DataFrame(mini), base], ignore_index=True)
    stats = {
        "rows_added": added,
        "jsonl_rows_considered": len(rows_in),
        "output_rows": len(aug),
    }
    logger.info("Simple-legit augment: added %s rows (jsonl considered=%s)", added, len(rows_in))
    return aug, stats
