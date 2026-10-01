"""Ingest brand-specific ``phish-*.txt`` / ``legit-*.txt`` files as **challenge / eval** data only.

These files are **not** the primary supervised training distribution. Use
:mod:`src.pipeline.kaggle_ingest` for primary binary training data.

Output: ``data/interim/challenge_normalized.csv`` with ``source_dataset=challenge_brand_files``.
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

_TAG_LINE = re.compile(r"^\[([a-z0-9_/-]+)\]\s*(\S+)\s*$", re.IGNORECASE)
_BRANDS = frozenset({"amazon", "google", "microsoft", "paypal"})


def _label_brand_from_filename(name: str) -> Tuple[int, str]:
    lower = name.lower()
    if lower.startswith("legit-") and lower.endswith(".txt"):
        stem = lower[6:-4]
        return 0, stem if stem in _BRANDS else stem or "unknown"
    if "legit" in lower and "legit-" not in lower:
        return 0, "legacy_legit"
    if lower.startswith("phish-") and lower.endswith(".txt"):
        stem = lower[6:-4]
        return 1, stem if stem in _BRANDS else stem or "unknown"
    return 1, "unknown"


def _parse_line(line: str) -> Tuple[str, str]:
    s = line.strip()
    if not s or s.startswith("#"):
        return "", ""
    m = _TAG_LINE.match(s)
    if m:
        return m.group(1).lower().replace("-", "_"), m.group(2).strip()
    return "uncategorized", s


def _read_txt(path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cat, url = _parse_line(line)
        if url:
            out.append((cat, url))
    return out


def ingest_challenge(raw_root: Optional[Path] = None) -> Path:
    ensure_layout()
    root = raw_root or raw_dir()
    rows: List[dict] = []
    for path in sorted(root.glob("*.txt")):
        if path.name.lower().startswith("readme"):
            continue
        name = path.name.lower()
        if not (name.startswith("phish-") or name.startswith("legit-")):
            continue
        lbl, brand = _label_brand_from_filename(path.name)
        for cat, u in _read_txt(path):
            rows.append(
                {
                    "url": u,
                    "source_file": path.name,
                    "source_dataset": "challenge_brand_files",
                    "source_brand_hint": brand,
                    "label": lbl,
                    "action_category": cat,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "url",
                "source_file",
                "source_dataset",
                "source_brand_hint",
                "label",
                "action_category",
            ]
        )
    else:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int).clip(0, 1)

    out_path = interim_dir() / "challenge_normalized.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "Challenge set: %s rows (legit=%s phish=%s) -> %s",
        len(df),
        int((df["label"] == 0).sum()) if len(df) else 0,
        int((df["label"] == 1).sum()) if len(df) else 0,
        out_path,
    )
    return out_path


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "ingest_challenge.log")
    ap = argparse.ArgumentParser(description="Ingest brand phish/legit txts as challenge/eval only.")
    ap.add_argument("--raw-dir", type=Path, default=None)
    args = ap.parse_args()
    ingest_challenge(args.raw_dir)


if __name__ == "__main__":
    main()
