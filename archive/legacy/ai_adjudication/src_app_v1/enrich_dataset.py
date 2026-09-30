"""Enrich an existing phishing URL dataset with brand/task-aware and lexical features.

This module is intentionally standalone and does NOT rebuild any dataset from
raw text files. It only reads an existing CSV and writes a NEW enriched CSV
that preserves all original columns and appends new feature columns.

CLI:
    python -m src.app_v1.enrich_dataset \
        --input data/raw/phishing_dataset.csv \
        --output data/processed/enriched_phishing_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

from .compare import _safe_domain


TRUSTED_LOOKUP = {
    ("microsoft", "login"): "https://login.microsoftonline.com",
    ("amazon", "login"): "https://www.amazon.com/ap/signin",
    ("paypal", "login"): "https://www.paypal.com/signin",
    ("google", "login"): "https://accounts.google.com",
}


URL_CANDIDATE_COLUMN_NAMES: Tuple[str, ...] = ("url", "URL", "link", "webpage")

# Base enrichment columns that are always added.
NEW_COLUMNS: Tuple[str, ...] = (
    "brand_guess",
    "task_guess",
    "reference_key",
    "trusted_reference_url",
    "trusted_reference_found",
    "is_reference_domain_match",
    "url_length",
    "num_dots",
    "has_ip_address",
    "has_at_symbol",
    "https_flag",
    "brand_in_domain",
    "brand_in_path",
    "invalid_url",
)

# Optional diagnostic columns that may be added if source columns are present.
OPTIONAL_COLUMNS: Tuple[str, ...] = (
    "brand_label_match",
)


def _detect_url_column(columns: Iterable[str]) -> Optional[str]:
    """Detect the URL column by name using simple heuristics."""
    cols = list(columns)
    lower_map = {c.lower(): c for c in cols}
    for candidate in URL_CANDIDATE_COLUMN_NAMES:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _normalize_url(raw: Optional[str]) -> Tuple[Optional[str], int]:
    """Normalize URL string for feature computation.

    Returns a tuple of (normalized_url, invalid_flag).

    - trims whitespace
    - lowercases the domain
    - removes a trailing slash from path-only URLs
    - flags URLs with missing scheme or netloc as invalid
    """
    if raw is None:
        return None, 1
    s = str(raw).strip()
    if not s:
        return None, 1

    parsed = urlparse(s)
    scheme = parsed.scheme or "http"
    netloc = (parsed.netloc or "").lower()
    invalid_flag = 0
    if not parsed.scheme or not parsed.netloc:
        invalid_flag = 1

    path = parsed.path or ""
    if path.endswith("/") and path != "/":
        path = path.rstrip("/")

    normalized = f"{scheme}://{netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"
    return normalized, invalid_flag


def _brand_from_url(url: Optional[str]) -> str:
    if not url:
        return "unknown"
    u = url.lower()
    if "microsoft" in u:
        return "microsoft"
    if "amazon" in u:
        return "amazon"
    if "paypal" in u:
        return "paypal"
    if "google" in u:
        return "google"
    return "unknown"


def _task_from_url(_url: Optional[str]) -> str:
    # For now, we assume login flows by default, as requested.
    return "login"


def _has_ip_address(url: str) -> int:
    """Detect if URL contains an IPv4 address."""
    import re

    if not url:
        return 0
    pattern = r"(?:\d{1,3}\.){3}\d{1,3}"
    return 1 if re.search(pattern, url) else 0


def _extract_domain_and_path(url: Optional[str]) -> Tuple[str, str]:
    if not url:
        return "", ""
    try:
        parsed = urlparse(url)
    except Exception:
        return "", ""
    domain = (parsed.netloc or "").lower()
    path = parsed.path or ""
    return domain, path


def _enrich_row(url_value: Optional[str]) -> dict:
    """Compute all requested enrichment fields for a single URL."""
    url_norm, invalid_url = _normalize_url(url_value)

    if not url_norm:
        # Invalid or missing URL; still return a full, JSON-serializable feature dict.
        return {
            "brand_guess": "unknown",
            "task_guess": "login",
            "reference_key": "unknown__login",
            "trusted_reference_url": None,
            "trusted_reference_found": False,
            "is_reference_domain_match": 0,
            "url_length": 0,
            "num_dots": 0,
            "has_ip_address": 0,
            "has_at_symbol": 0,
            "https_flag": 0,
            "brand_in_domain": 0,
            "brand_in_path": 0,
            "invalid_url": int(invalid_url),
        }

    brand_guess = _brand_from_url(url_norm)
    task_guess = _task_from_url(url_norm)
    reference_key = f"{brand_guess}__{task_guess}"

    trusted_reference_url = TRUSTED_LOOKUP.get((brand_guess, task_guess))
    trusted_reference_found = trusted_reference_url is not None

    # Domain match logic
    domain_url = _safe_domain(url_norm)
    domain_ref = _safe_domain(trusted_reference_url) if trusted_reference_url else ""
    is_reference_domain_match = int(bool(domain_url and domain_ref and domain_url == domain_ref))

    # Lexical features
    url_length = len(url_norm)
    num_dots = url_norm.count(".")
    has_at_symbol = 1 if "@" in url_norm else 0
    https_flag = 1 if url_norm.lower().startswith("https://") else 0
    has_ip_address = _has_ip_address(url_norm)

    domain, path = _extract_domain_and_path(url_norm)
    brand_in_domain = 1 if brand_guess != "unknown" and brand_guess in domain else 0
    brand_in_path = 1 if brand_guess != "unknown" and brand_guess in path.lower() else 0

    return {
        "brand_guess": brand_guess,
        "task_guess": task_guess,
        "reference_key": reference_key,
        "trusted_reference_url": trusted_reference_url,
        "trusted_reference_found": bool(trusted_reference_found),
        "is_reference_domain_match": is_reference_domain_match,
        "url_length": url_length,
        "num_dots": num_dots,
        "has_ip_address": has_ip_address,
        "has_at_symbol": has_at_symbol,
        "https_flag": https_flag,
        "brand_in_domain": brand_in_domain,
        "brand_in_path": brand_in_path,
        "invalid_url": int(invalid_url),
    }


def enrich_dataset(input_path: str, output_path: str) -> None:
    """Load CSV, enrich it in memory, and write a new enriched CSV."""
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise ValueError(f"Input dataset is empty: {in_path}")

    url_col = _detect_url_column(df.columns)
    if not url_col:
        raise ValueError(
            "Could not detect URL column. "
            "Tried: url, URL, link, webpage (case-insensitive)."
        )

    # Compute enriched features row-wise, preserving all original columns.
    features = df[url_col].apply(_enrich_row)
    features_df = pd.DataFrame(list(features))

    # Optional diagnostic: brand_label_match iff an original 'brand' column exists.
    brand_col: Optional[str] = None
    for c in df.columns:
        if c.lower() == "brand":
            brand_col = c
            break
    if brand_col is not None:
        brand_guess_series = features_df["brand_guess"].astype(str).str.strip().str.lower()
        brand_label_series = df[brand_col].astype(str).str.strip().str.lower()
        features_df["brand_label_match"] = (brand_guess_series == brand_label_series).astype(int)

    # Ensure we only append new columns (avoid clobbering any existing ones).
    for col in list(NEW_COLUMNS) + list(OPTIONAL_COLUMNS):
        if col in df.columns:
            raise ValueError(
                f"Column '{col}' already exists in input dataset; "
                "refusing to overwrite for safety."
            )

    enriched = pd.concat([df, features_df], axis=1)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)

    # Print summary and diagnostics for quick inspection.
    print(f"Enriched dataset written to: {out_path}")
    print(f"Total rows: {len(enriched)}")

    # Validation summary metrics.
    print("\n=== Validation summary ===")
    print("Rows by brand_guess:")
    print(enriched["brand_guess"].value_counts(dropna=False))
    print("\nRows by task_guess:")
    print(enriched["task_guess"].value_counts(dropna=False))
    print("\ntrusted_reference_found counts:")
    print(enriched["trusted_reference_found"].value_counts(dropna=False))
    print("\nis_reference_domain_match counts:")
    print(enriched["is_reference_domain_match"].value_counts(dropna=False))
    print("\ninvalid_url counts:")
    print(enriched["invalid_url"].value_counts(dropna=False))

    # Console preview: only selected columns, truncated URLs (CSV remains full).
    print("\n=== Sample preview (first 5 rows) ===")
    preview_cols: List[str] = []
    if url_col in enriched.columns:
        preview_cols.append(url_col)
    if "brand" in enriched.columns:
        preview_cols.append("brand")
    preview_cols.extend(
        [
            "brand_guess",
            "task_guess",
            "trusted_reference_found",
            "is_reference_domain_match",
            "invalid_url",
        ]
    )
    if "brand_label_match" in enriched.columns:
        preview_cols.append("brand_label_match")

    preview_cols = [c for c in preview_cols if c in enriched.columns]
    preview_df = enriched.loc[:, preview_cols].head(5).copy()
    if url_col in preview_df.columns:
        preview_df[url_col] = preview_df[url_col].astype(str).str.slice(0, 80)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(preview_df.to_string(index=False))

    # Suspicious diagnostic rows.
    print("\n=== Suspicious rows (up to 10) ===")
    suspicious_mask = (
        (enriched["brand_guess"] == "unknown")
        | (~enriched["trusted_reference_found"])
        | (enriched["invalid_url"] == 1)
    )
    suspicious = enriched.loc[suspicious_mask].head(10)
    if not suspicious.empty:
        diag_cols: List[str] = []
        if url_col in suspicious.columns:
            diag_cols.append(url_col)
        if "brand" in suspicious.columns:
            diag_cols.append("brand")
        diag_cols.extend(
            [
                "brand_guess",
                "task_guess",
                "trusted_reference_found",
                "is_reference_domain_match",
                "invalid_url",
            ]
        )
        if "brand_label_match" in suspicious.columns:
            diag_cols.append("brand_label_match")
        diag_cols = [c for c in diag_cols if c in suspicious.columns]
        suspicious_view = suspicious.loc[:, diag_cols].copy()
        if url_col in suspicious_view.columns:
            suspicious_view[url_col] = suspicious_view[url_col].astype(str).str.slice(0, 80)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(suspicious_view.to_string(index=False))
    else:
        print("No suspicious rows matched the diagnostic criteria.")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich phishing_dataset.csv with brand/task-aware and lexical features.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to existing phishing_dataset.csv (will not be modified).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the NEW enriched CSV.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    enrich_dataset(args.input, args.output)


if __name__ == "__main__":
    # Support both `python enrich_dataset.py` and `python -m src.app_v1.enrich_dataset`.
    main(sys.argv[1:])

