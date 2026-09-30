"""Rebuild Phase 1 check: re-run the documented 50K pipeline with the Phase 1 fixes and record
what changed. This is a progress check, not the final evaluation (that is Phase 3).

Writes metrics/results/10_phase1_check.json. Artifacts go to metrics/_work/phase1/ (gitignored).
"""

from __future__ import annotations

import json
import os
import time

from common import RAW_KAGGLE_CSV, SEED, WORK, seed_everything, write_result

P1 = WORK / "phase1"
os.environ["PHISH_DATA_DIR"] = str(P1 / "data")
os.environ["PHISH_OUTPUTS_DIR"] = str(P1 / "outputs")
os.environ["PHISH_LOGS_DIR"] = str(P1 / "logs")
from common import isolate_env  # noqa: E402

isolate_env()
seed_everything()

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from src.pipeline.evaluation_sets import load_official_brand_rows, load_phishstats_rows, load_url_suites  # noqa: E402
from src.pipeline.logging_util import setup_logging  # noqa: E402
from src.pipeline.run_kaggle_pipeline import run_kaggle_pipeline  # noqa: E402


def main() -> None:
    setup_logging(P1 / "logs" / "phase1.log")
    t0 = time.perf_counter()
    run_kaggle_pipeline(csv_path=RAW_KAGGLE_CSV, seed=SEED, sample_size=50000, use_fresh_data=False,
                        primary_selection="validated", write_primary_artifact=True, enrich_resume=False,
                        checkpoint_every=10**9)
    runtime = time.perf_counter() - t0
    out = P1 / "outputs"
    rep = lambda n: json.loads((out / "reports" / n).read_text())  # noqa: E731
    metrics = json.loads((out / "metrics" / "metrics.json").read_text())
    bundle = joblib.load(out / "models" / "layer1_bundle.joblib")

    from src.app_v1.analyze_dashboard import build_dashboard_analysis
    from src.app_v1.ml_layer1 import predict_layer1

    def l1_flags(urls):
        return [bool(predict_layer1(u).get("predicted_phishing")) for u in urls]

    off_test = [r["url"] for r in load_official_brand_rows(split="test")]
    ph = [r["url"] for r in load_phishstats_rows()]
    suites = {}
    for name, urls in load_url_suites().items():
        v = [build_dashboard_analysis(u, reinforcement=False)[0]["verdict"]["verdict_3way"] for u in urls]
        suites[name] = {"n": len(urls), "verdicts": {k: v.count(k) for k in set(v)}}
    off_dash = [build_dashboard_analysis(u, reinforcement=False)[0]["verdict"]["verdict_3way"] for u in off_test]

    write_result("10_phase1_check", {
        "command": "python metrics/scripts/10_phase1_check.py",
        "note": "Progress check after Phase 1 fixes; 50K stratified sample, seed 42, ML-only dashboard path.",
        "pipeline_runtime_seconds": round(runtime, 1),
        "evaluation_exclusions": rep("evaluation_exclusions.json"),
        "split": rep("split_leak_safe_stats.json"),
        "training_config": {k: rep("training_config.json").get(k) for k in (
            "n_fit_rows", "n_validation_rows", "validation_split", "feature_parity_check",
            "layer1_dropped_from_training_features", "layer1_primary_selection_policy")},
        "n_features": len(bundle["feature_columns"]),
        "models_test_threshold_0_5": [{k: v for k, v in m.items() if k != "confusion_matrix" or True} for m in metrics],
        "selected_primary": bundle["model_name"],
        "bundle": {k: bundle[k] for k in ("selection_policy", "train_csv_sha256", "n_fit_rows", "n_validation_rows", "n_test_rows", "created_utc")},
        "official_brand_test_half": {"n": len(off_test), "layer1_fp_rate": float(np.mean(l1_flags(off_test))),
                                     "dashboard_verdicts": {k: off_dash.count(k) for k in set(off_dash)}},
        "phishstats": {"n": len(ph), "layer1_recall": float(np.mean(l1_flags(ph)))},
        "url_suites_dashboard_ml_only": suites,
    })


if __name__ == "__main__":
    main()
