"""Legacy ingest: all ``*.txt`` / ``*.csv`` under ``data/raw/`` → ``interim/normalized.csv``.

**Primary supervised training** should use the Kaggle binary dataset instead
(see :mod:`src.pipeline.kaggle_ingest` and ``docs/ARCHITECTURE.md``).

Brand-specific ``phish-*`` / ``legit-*`` files are **challenge/eval** only when
you need a curated mix; prefer :mod:`src.pipeline.ingest_challenge` for that.

Labels (internal):
  - phishing: label = 1
  - legitimate: label = 0

Optional tagged lines::
  [login_auth] https://example.com/path
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.pipeline.paths import ensure_layout, interim_dir, raw_dir

logger = logging.getLogger(__name__)

# [category] URL — category is lowercase snake_case recommended
_TAG_LINE = re.compile(r"^\[([a-z0-9_/-]+)\]\s*(\S+)\s*$", re.IGNORECASE)

_BRANDS = frozenset({"amazon", "google", "microsoft", "paypal"})


def _label_and_brand_from_filename(name: str) -> Tuple[int, str]:
    """Return (label, source_brand_hint). Phishing = 1, legit = 0."""
    lower = name.lower()
    if lower.startswith("legit-") and lower.endswith(".txt"):
        stem = lower[6:-4]
        if stem in _BRANDS:
            return 0, stem
        return 0, stem if stem else "unknown"
    if "legit" in lower and "legit-" not in lower:
        # Legacy e.g. legit-urls-sample.txt
        return 0, _brand_from_heuristic(lower)
    if lower.startswith("phish-") and lower.endswith(".txt"):
        stem = lower[6:-4]
        return 1, stem if stem in _BRANDS else _brand_from_heuristic(lower)
    # Default: treat as phishing unless filename says legit
    return 1, _brand_from_heuristic(lower)


def _brand_from_heuristic(name: str) -> str:
    for b in ("amazon", "google", "microsoft", "paypal"):
        if b in name:
            return b
    return "unknown"


def _parse_txt_line(line: str) -> Tuple[str, str]:
    """Return (action_category, url_or_empty). Empty skips line."""
    s = line.strip()
    if not s or s.startswith("#"):
        return "", ""
    m = _TAG_LINE.match(s)
    if m:
        cat = m.group(1).lower().replace("-", "_")
        url = m.group(2).strip()
        return cat, url
    return "uncategorized", s


def _read_txt_urls(path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cat, url = _parse_txt_line(line)
        if url:
            out.append((cat, url))
    return out


def _read_csv_urls(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    url_col = None
    for key in ("url", "link", "uri", "website", "webpage"):
        if key in cols:
            url_col = cols[key]
            break
    if url_col is None:
        raise ValueError(f"No URL column found in {path}")
    out = pd.DataFrame(
        {
            "url": df[url_col].astype(str),
        }
    )
    if "label" in cols:
        out["label"] = pd.to_numeric(df[cols["label"]], errors="coerce").fillna(1).astype(int)
    else:
        lbl, br = _label_and_brand_from_filename(path.name)
        out["label"] = lbl
    if "source_brand_hint" in cols:
        out["source_brand_hint"] = df[cols["source_brand_hint"]].astype(str)
    else:
        _, br = _label_and_brand_from_filename(path.name)
        out["source_brand_hint"] = br
    if "action_category" in cols:
        out["action_category"] = df[cols["action_category"]].astype(str)
    else:
        out["action_category"] = "uncategorized"
    out["source_file"] = path.name
    return out


def ingest(raw_root: Optional[Path] = None) -> Path:
    ensure_layout()
    root = raw_root or raw_dir()
    rows: List[dict] = []
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        logger.warning("Raw directory empty or missing: %s", root)

    for path in sorted(root.glob("*.txt")):
        if path.name.lower().startswith("readme"):
            continue
        label, brand = _label_and_brand_from_filename(path.name)
        for cat, u in _read_txt_urls(path):
            rows.append(
                {
                    "url": u,
                    "source_file": path.name,
                    "source_brand_hint": brand,
                    "label": label,
                    "action_category": cat,
                }
            )

    for path in sorted(root.glob("*.csv")):
        try:
            cdf = _read_csv_urls(path)
            for _, r in cdf.iterrows():
                rows.append(r.to_dict())
        except Exception as e:
            logger.error("Skip CSV %s: %s", path, e)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "url",
                "source_file",
                "source_brand_hint",
                "label",
                "action_category",
            ]
        )
    else:
        # Enforce int labels 0/1
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int).clip(0, 1)

    out_path = interim_dir() / "normalized.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "Wrote %s rows (legit=%s phish=%s) -> %s",
        len(df),
        int((df["label"] == 0).sum()) if len(df) else 0,
        int((df["label"] == 1).sum()) if len(df) else 0,
        out_path,
    )
    return out_path


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "ingest.log")
    p = argparse.ArgumentParser(description="Ingest raw phishing datasets.")
    p.add_argument("--raw-dir", type=Path, default=None)
    args = p.parse_args()
    ingest(args.raw_dir)


if __name__ == "__main__":
    main()
