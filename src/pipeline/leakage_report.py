"""Audit train/test CSVs for URL overlap, group overlap, and excluded metadata columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

from src.pipeline.paths import ensure_layout, processed_dir, reports_dir
from src.pipeline.split_leak_safe import ensure_group_column
from src.pipeline.train import BASE_EXCLUDE_FROM_X, FETCH_PROXY_FEATURES


def build_leakage_report(
    train_csv: Path | None = None,
    test_csv: Path | None = None,
) -> Dict[str, Any]:
    ensure_layout()
    tr = train_csv or (processed_dir() / "train.csv")
    te = test_csv or (processed_dir() / "test.csv")
    train = pd.read_csv(tr, dtype=str, low_memory=False)
    test = pd.read_csv(te, dtype=str, low_memory=False)

    c_tr = set(train["canonical_url"].astype(str)) if "canonical_url" in train.columns else set()
    c_te = set(test["canonical_url"].astype(str)) if "canonical_url" in test.columns else set()
    g_tr = set(ensure_group_column(train)) if len(train) else set()
    g_te = set(ensure_group_column(test)) if len(test) else set()

    metadata_excluded: List[str] = sorted(
        BASE_EXCLUDE_FROM_X | FETCH_PROXY_FEATURES | {"_split", "_leak_group"}
    )

    report: Dict[str, Any] = {
        "train_csv": str(tr),
        "test_csv": str(te),
        "train_rows": len(train),
        "test_rows": len(test),
        "canonical_url_overlap": sorted(c_tr & c_te),
        "canonical_url_overlap_count": len(c_tr & c_te),
        "group_key_overlap_count": len(g_tr & g_te),
        "metadata_columns_excluded_from_model_features": metadata_excluded,
        "notes": [
            "Model training must exclude metadata columns listed above.",
            "Group overlap should be 0 when using split_leak_safe.StratifiedGroupKFold.",
        ],
    }
    if report["canonical_url_overlap_count"] or report["group_key_overlap_count"]:
        report["leakage_risk"] = "HIGH"
    else:
        report["leakage_risk"] = "LOW"
    out = reports_dir() / "leakage_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "leakage_report.log")
    ap = argparse.ArgumentParser(description="Train/test leakage audit.")
    ap.add_argument("--train", type=Path, default=None)
    ap.add_argument("--test", type=Path, default=None)
    args = ap.parse_args()
    r = build_leakage_report(args.train, args.test)
    print(json.dumps({k: r[k] for k in ("leakage_risk", "canonical_url_overlap_count", "group_key_overlap_count")}, indent=2))


if __name__ == "__main__":
    main()
