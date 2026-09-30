"""Paths and loaders for curated evaluation URL lists (hard legit, suites)."""

from __future__ import annotations

import json
from pathlib import Path

from phishguard.paths import project_root
from typing import Any, Dict, List, Optional

from phishguard.data.augment import DEFAULT_SIMPLE_LEGIT_JSONL, load_simple_legit_rows

# Repo-root-relative defaults (resolve from this file).
_REPO_ROOT = project_root()
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


# --- Evaluation-only sets and training exclusions (rebuild Phase 1) ---------------------------
DEFAULT_OFFICIAL_BRAND_JSONL = _REPO_ROOT / "data" / "evaluation" / "official_brand_urls.jsonl"
DEFAULT_PHISHSTATS_JSONL = _REPO_ROOT / "data" / "evaluation" / "phishstats_urls.jsonl"


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_official_brand_rows(split: Optional[str] = None, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """298 curated official brand URLs (legitimate). ``split`` is 'val' (model selection) or 'test' (reporting)."""
    rows = _read_jsonl(path or DEFAULT_OFFICIAL_BRAND_JSONL)
    return [r for r in rows if split is None or r.get("split") == split]


def load_phishstats_rows(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """PhishStats phishing URLs used only for evaluation."""
    return _read_jsonl(path or DEFAULT_PHISHSTATS_JSONL)


def evaluation_exclusions() -> Dict[str, set]:
    """What must never appear in training so evaluation stays out-of-sample.

    * ``domains``: registered domains of every LEGITIMATE evaluation URL (url_suites legit buckets,
      hard_legit, official brand URLs, and the audit URLs the old composite rule scored). All rows
      on these domains are removed, so "does it flag google.com" is asked about an unseen site.
    * ``hosts``: exact hostnames of every PHISHING evaluation URL (url_suites phish buckets,
      PhishStats). Matched by host, not registered domain, because many live on shared hosting
      (github.io, netlify.app) and dropping those whole domains would gut the training data.
    """
    from phishguard.urls.safe import leak_safe_group_key, safe_hostname
    from phishguard.models.train import LAYER1_OFFICIAL_HTTPS_AUDIT_URLS
    from phishguard.urls.normalize import canonical_url

    legit: List[str] = []
    phish: List[str] = []
    for name, urls in load_url_suites().items():
        (legit if "legit" in name.lower() else phish).extend(urls)
    legit += [r["url"] for r in load_hard_legit_rows()]
    legit += [r["url"] for r in load_official_brand_rows()]
    legit += list(LAYER1_OFFICIAL_HTTPS_AUDIT_URLS)
    phish += [r["url"] for r in load_phishstats_rows()]
    domains = {leak_safe_group_key(u)[0] for u in legit}
    hosts = {safe_hostname(canonical_url(u)[0])[0] for u in phish}
    domains.discard("")
    hosts.discard("")
    return {"domains": domains, "hosts": hosts}


def drop_evaluation_rows(df, url_col: str = "canonical_url"):
    """Return (kept_df, stats). Removes rows that would put evaluation URLs into training."""
    from phishguard.urls.safe import leak_safe_group_key, safe_hostname
    from phishguard.urls.normalize import canonical_url

    ex = evaluation_exclusions()
    urls = df[url_col].fillna("").astype(str)
    dom = urls.map(lambda u: leak_safe_group_key(u)[0])
    host = urls.map(lambda u: safe_hostname(canonical_url(u)[0])[0])
    by_dom = dom.isin(ex["domains"])
    by_host = host.isin(ex["hosts"])
    drop = by_dom | by_host
    stats = {
        "rows_in": int(len(df)),
        "rows_dropped_legit_eval_domains": int(by_dom.sum()),
        "rows_dropped_phish_eval_hosts": int((by_host & ~by_dom).sum()),
        "rows_out": int((~drop).sum()),
        "n_excluded_domains": len(ex["domains"]),
        "n_excluded_hosts": len(ex["hosts"]),
    }
    return df.loc[~drop.values].copy(), stats
