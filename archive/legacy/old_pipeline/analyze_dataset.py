"""Analyze enriched dataset: imbalance, trivial separability, leakage risk."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.pipeline.paths import ensure_layout, processed_dir, reports_dir

logger = logging.getLogger(__name__)


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _class_balance(df: pd.DataFrame) -> Dict[str, Any]:
    vc = df["label"].astype(int).value_counts()
    n0 = int(vc.get(0, 0))
    n1 = int(vc.get(1, 0))
    tot = max(n0 + n1, 1)
    return {
        "n_legit_0": n0,
        "n_phish_1": n1,
        "ratio_phish": round(n1 / tot, 4),
        "imbalance_ratio": round(max(n1, n0) / max(min(n1, n0), 1), 4),
    }


def _fetch_vs_label(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "page_fetch_success" not in df.columns:
        return {"note": "no page_fetch_success column"}
    y = df["label"].astype(int)
    pfs = _safe_numeric(df, "page_fetch_success").fillna(0).astype(int)
    ct = pd.crosstab(y, pfs, margins=True)
    out["crosstab_label_vs_page_fetch_success"] = {str(k): {str(c): int(v) for c, v in row.items()} for k, row in ct.iterrows()}
    ph = y == 1
    if ph.sum() > 0:
        fail_rate_phish = float((pfs[ph] == 0).mean())
        out["phish_fetch_fail_rate"] = round(fail_rate_phish, 4)
    if (~ph).sum() > 0:
        fail_rate_legit = float((pfs[~ph] == 0).mean())
        out["legit_fetch_fail_rate"] = round(fail_rate_legit, 4)
    return out


def _univariate_separability(df: pd.DataFrame, top_k: int = 25) -> List[Dict[str, Any]]:
    y = df["label"].astype(int).values
    rows: List[Dict[str, Any]] = []
    if len(np.unique(y)) < 2:
        return rows
    for c in df.columns:
        if c in {"label", "url", "canonical_url", "final_url", "source_file", "fetch_error"}:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < max(10, 0.2 * len(df)):
            continue
        v0 = s[y == 0].dropna()
        v1 = s[y == 1].dropna()
        if len(v0) < 3 or len(v1) < 3:
            continue
        m0, m1 = float(v0.mean()), float(v1.mean())
        pooled = np.sqrt((v0.var() + v1.var()) / 2 + 1e-9)
        cohen_d = abs(m1 - m0) / pooled
        rows.append({"feature": c, "mean_legit": m0, "mean_phish": m1, "cohen_d": round(float(cohen_d), 4)})
    rows.sort(key=lambda x: -x["cohen_d"])
    return rows[:top_k]


def _label_correlation(df: pd.DataFrame, top_k: int = 20) -> List[Dict[str, Any]]:
    y = df["label"].astype(float).values
    out: List[Dict[str, Any]] = []
    for c in df.columns:
        if c == "label":
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < max(10, 0.2 * len(df)):
            continue
        m = np.isfinite(s.values) & np.isfinite(y)
        if m.sum() < 10:
            continue
        r = np.corrcoef(s.values[m], y[m])[0, 1]
        if np.isnan(r):
            continue
        out.append({"feature": c, "pearson_r_label": round(float(r), 4)})
    out.sort(key=lambda x: -abs(x["pearson_r_label"]))
    return out[:top_k]


def _leakage_notes(fetch_block: Dict[str, Any], top_corr: List[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    frp = fetch_block.get("phish_fetch_fail_rate")
    frl = fetch_block.get("legit_fetch_fail_rate")
    if frp is not None and frl is not None and frp > frl + 0.15:
        notes.append(
            "Phishing URLs fail HTTP fetch more often than legit; models may learn fetch-failure instead of content."
        )
    for row in top_corr[:8]:
        if row["feature"] in {
            "page_fetch_success",
            "fetch_features_missing",
            "fetch_error_flag",
            "http_status",
            "html_features_missing",
            "html_length",
        }:
            notes.append(
                f"High label correlation on `{row['feature']}` (r={row['pearson_r_label']}) — possible leakage / artifact."
            )
    return notes


def analyze(enriched_csv: Path | None = None) -> Path:
    ensure_layout()
    path = enriched_csv or (processed_dir() / "enriched.csv")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    report: Dict[str, Any] = {
        "source": str(path),
        "n_rows": len(df),
        "class_balance": _class_balance(df),
        "fetch_vs_label": _fetch_vs_label(df),
        "top_univariate_separation_cohen_d": _univariate_separability(df),
        "top_label_correlations": _label_correlation(df),
        "leakage_notes": [],
    }
    report["leakage_notes"] = _leakage_notes(report["fetch_vs_label"], report["top_label_correlations"])

    out = reports_dir() / "dataset_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    print(json.dumps({k: v for k, v in report.items() if k != "top_univariate_separation_cohen_d"}, indent=2))
    print("\nTop Cohen's d (trivial separation risk):")
    for row in report["top_univariate_separation_cohen_d"][:15]:
        print(f"  {row['feature']}: d={row['cohen_d']} legit_mean={row['mean_legit']:.4g} phish_mean={row['mean_phish']:.4g}")
    return out


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "analyze_dataset.log")
    p = argparse.ArgumentParser(description="Dataset realism / leakage analysis.")
    p.add_argument("--input", type=Path, default=None)
    args = p.parse_args()
    analyze(args.input)


if __name__ == "__main__":
    main()
