"""Step 1: run the repo's own documented Layer-1 pipeline end to end and record dataset stats + runtime.

Documented default (docs/DATASET_SETUP.md): 50,000-row stratified sample of the deduplicated Kaggle
dump, seed 42, Kaggle-only (no fresh data), composite primary-selection rule.
All artifacts go to metrics/_work/ (see common.isolate_env).
"""

from __future__ import annotations

import argparse
import json
import time

from common import RAW_KAGGLE_CSV, SEED, WORK, isolate_env, seed_everything, sha256, write_result

isolate_env()
seed_everything()

import pandas as pd  # noqa: E402

from phishguard.logging_util import setup_logging  # noqa: E402
from phishguard.pipelines.kaggle import run_kaggle_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=50000)
    args = ap.parse_args()

    setup_logging(WORK / "logs" / "verify_pipeline.log")
    raw = pd.read_csv(RAW_KAGGLE_CSV, dtype=str)
    raw_stats = {
        "path": "data/raw/kaggle/phishing_and_legitimate_urls.csv",
        "sha256": sha256(RAW_KAGGLE_CSV),
        "rows": int(len(raw)),
        "columns": list(raw.columns),
        "n_columns": int(raw.shape[1]),
        "status_counts_raw_kaggle_convention": raw["status"].value_counts().to_dict(),
        "note": "Kaggle convention: status 1 = legitimate, 0 = phishing. Internal label: 1 = phishing.",
        "date_range": "not available (dataset has only url,status columns; no timestamps)",
    }

    t0 = time.perf_counter()
    run_kaggle_pipeline(
        csv_path=RAW_KAGGLE_CSV,
        seed=SEED,
        sample_size=args.sample_size,
        use_fresh_data=False,
        primary_selection="composite",
        write_primary_artifact=True,
        enrich_resume=False,
    )
    runtime_s = time.perf_counter() - t0

    out = WORK / "outputs"
    manifest = json.loads((out / "reports" / "kaggle_sample_manifest.json").read_text())
    clean_stats = json.loads((out / "reports" / "kaggle_clean_stats.json").read_text())
    leak = json.loads((out / "reports" / "leakage_audit.json").read_text())
    split = json.loads((out / "reports" / "split_leak_safe_stats.json").read_text())
    metrics = json.loads((out / "metrics" / "metrics.json").read_text())
    tcfg = json.loads((out / "reports" / "training_config.json").read_text())

    write_result(
        "01_pipeline",
        {
            "command": f"python metrics/scripts/01_pipeline.py --sample-size {args.sample_size}",
            "pipeline_entrypoint": "phishguard.pipelines.kaggle.run_kaggle_pipeline (repo code, unmodified)",
            "end_to_end_runtime_seconds": round(runtime_s, 1),
            "raw_dataset": raw_stats,
            "dedup": {k: v for k, v in clean_stats.items() if k not in ("input", "output")},
            "sample_manifest": {k: v for k, v in manifest.items() if k not in ("full_deduplicated_csv", "sampled_cleaned_csv")},
            "split": split,
            "leakage_audit": {k: v for k, v in leak.items() if k not in ("train_csv", "test_csv")},
            "n_features_used": len(tcfg.get("feature_columns_used", [])),
            "feature_columns_used": tcfg.get("feature_columns_used", []),
            "repo_train_py_metrics_threshold_0_5": metrics,
            "calibration": {k: v for k, v in (tcfg.get("layer1_calibration_report") or {}).items() if k != "path"},
        },
    )


if __name__ == "__main__":
    main()
