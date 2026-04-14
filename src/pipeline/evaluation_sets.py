"""Paths and loaders for curated evaluation URL lists (hard legit, suites)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.simple_legit_augment import DEFAULT_SIMPLE_LEGIT_JSONL, load_simple_legit_rows

# Repo-root-relative defaults (resolve from this file).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARD_LEGIT_JSONL = _REPO_ROOT / "data" / "evaluation" / "hard_legit_urls.jsonl"
DEFAULT_URL_SUITES_JSON = _REPO_ROOT / "data" / "evaluation" / "url_suites.json"

__all__ = [
    "DEFAULT_HARD_LEGIT_JSONL",
    "DEFAULT_SIMPLE_LEGIT_JSONL",
    "DEFAULT_URL_SUITES_JSON",
    "load_hard_legit_rows",
    "load_simple_legit_rows",
    "load_url_suites",
]


def load_hard_legit_rows(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or DEFAULT_HARD_LEGIT_JSONL
    if not p.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_url_suites(path: Optional[Path] = None) -> Dict[str, List[str]]:
    p = path or DEFAULT_URL_SUITES_JSON
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k: list(v) for k, v in data.items() if isinstance(v, list)}
