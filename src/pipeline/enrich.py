"""Resumable feature enrichment (run inside container)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from src.pipeline.features.behavior_playwright import merge_behavior
from src.pipeline.features.dns_features import extract_dns_features
from src.pipeline.features.hosting_features import extract_hosting_features
from src.pipeline.features.html_dom import extract_dom_features, fetch_html_http
from src.pipeline.features.semantic_text import extract_semantic_features
from src.pipeline.features.url_features import extract_url_features
from src.pipeline.layer1_features import extract_layer1_features
from src.pipeline.paths import ensure_layout, interim_dir, processed_dir

logger = logging.getLogger(__name__)


def _hostname(url: str) -> str:
    h, _ = safe_hostname(url)
    return h


def enrich_row(
    canonical_url: str,
    *,
    use_playwright: bool,
    artifact_root: Optional[Path],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"canonical_url": canonical_url}
    row.update(extract_url_features(canonical_url))
    row.update(extract_hosting_features(canonical_url))
    host = _hostname(canonical_url)
    row.update(extract_dns_features(host))

    fetch_out, html = fetch_html_http(canonical_url)
    row.update(fetch_out)
    dom = extract_dom_features(canonical_url, fetch_out, html)
    title = dom.pop("_title_text", "")
    visible = dom.pop("_visible_text", "")
    row.update(dom)
    row.update(extract_semantic_features(title, visible))

    http_ok = int(bool(row.get("page_fetch_success")))
    row.update(
        merge_behavior(
            http_ok,
            int(row.get("html_features_missing", 1)),
            canonical_url,
            use_playwright=use_playwright,
            artifact_root=artifact_root,
        )
    )

    row["url_features_missing"] = 0
    row["hosting_features_missing"] = int(not host)
    row["dns_features_missing"] = int(row.get("dns_missing", 0))
    row["fetch_features_missing"] = int(not http_ok)
    row["fetch_error_flag"] = int(bool(row.get("fetch_error")))
    return row


def enrich_row_layer1_only(canonical_url: str, *, use_dns: bool = False) -> Dict[str, Any]:
    """URL + hosting (+ optional DNS). No HTTP body fetch, no Playwright."""
    return extract_layer1_features(canonical_url, use_dns=use_dns)


def _load_feature_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, low_memory=False)
    return pd.DataFrame()


def _save_feature_table(path: Path, acc: pd.DataFrame) -> None:
    acc.to_csv(path, index=False)


def enrich(
    cleaned_csv: Optional[Path] = None,
    *,
    limit: Optional[int] = None,
    resume: bool = True,
    use_playwright: bool = False,
    checkpoint_every: Optional[int] = None,
    artifact_root: Optional[Path] = None,
    layer1_only: bool = False,
    layer1_use_dns: bool = False,
    output_csv: Optional[Path] = None,
    checkpoint_name: str = "enriched_features.csv",
) -> Path:
    ensure_layout()
    ck_every = checkpoint_every if checkpoint_every is not None else (400 if layer1_only else 25)
    in_path = cleaned_csv or (processed_dir() / "cleaned.csv")
    base = pd.read_csv(in_path, dtype=str, low_memory=False)
    if base.empty:
        out_path = processed_dir() / "enriched.csv"
        base.to_csv(out_path, index=False)
        return out_path

    ck_path = interim_dir() / checkpoint_name
    if not resume and ck_path.exists():
        ck_path.unlink()
    acc = _load_feature_table(ck_path) if resume else pd.DataFrame()
    done: set = set()
    if not acc.empty and "canonical_url" in acc.columns:
        done = set(acc["canonical_url"].astype(str))

    # Rows to fetch this run (full dataset unless --limit for debug).
    work = base if limit is None else base.head(int(limit))
    batch: List[Dict[str, Any]] = []
    for _, r in tqdm(work.iterrows(), total=len(work), desc="enrich"):
        key = str(r.get("canonical_url", ""))
        if not key or key in done:
            continue
        try:
            if layer1_only:
                feats = enrich_row_layer1_only(key, use_dns=layer1_use_dns)
            else:
                feats = enrich_row(key, use_playwright=use_playwright, artifact_root=artifact_root)
        except Exception as e:
            logger.exception("enrich_row failed for %s: %s", key, e)
            feats = {"canonical_url": key, "enrich_fatal_error": str(e)}
        batch.append(feats)
        done.add(key)
        if len(batch) >= ck_every:
            acc = pd.concat([acc, pd.DataFrame(batch)], ignore_index=True)
            acc = acc.drop_duplicates(subset=["canonical_url"], keep="last")
            _save_feature_table(ck_path, acc)
            batch.clear()

    if batch:
        acc = pd.concat([acc, pd.DataFrame(batch)], ignore_index=True)
        acc = acc.drop_duplicates(subset=["canonical_url"], keep="last")
        _save_feature_table(ck_path, acc)

    merge_base = base if limit is None else work
    if acc.empty:
        out = merge_base.copy()
    else:
        out = merge_base.merge(acc, on="canonical_url", how="left")
    out_path = output_csv or (processed_dir() / "enriched.csv")
    out.to_csv(out_path, index=False)
    logger.info(
        "Wrote enriched -> %s rows=%s (layer1_only=%s checkpoint_every=%s)",
        out_path,
        len(out),
        layer1_only,
        ck_every,
    )
    return out_path


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir, outputs_dir

    setup_logging(logs_dir() / "enrich.log")
    p = argparse.ArgumentParser(description="Feature enrichment (container-safe).")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--playwright", action="store_true")
    p.add_argument("--artifacts", type=Path, default=None)
    p.add_argument(
        "--layer1-only",
        action="store_true",
        help="URL + hosting (+ optional DNS) only; no HTTP HTML fetch or Playwright.",
    )
    p.add_argument(
        "--layer1-use-dns",
        action="store_true",
        help="With --layer1-only, run DNS lookups (slower for large batches).",
    )
    p.add_argument("--output", type=Path, default=None, help="Output enriched CSV path")
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default="enriched_features.csv",
        help="Checkpoint filename under data/interim/",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Rows between checkpoint writes (default: 400 layer1-only, 25 full enrich).",
    )
    args = p.parse_args()
    art = args.artifacts or (outputs_dir() / "enrich_artifacts")
    enrich(
        args.input,
        limit=args.limit,
        resume=not args.no_resume,
        use_playwright=args.playwright,
        artifact_root=art,
        layer1_only=args.layer1_only,
        layer1_use_dns=args.layer1_use_dns,
        output_csv=args.output,
        checkpoint_name=args.checkpoint_name,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
