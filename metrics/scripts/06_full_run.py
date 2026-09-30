"""Step 6: the full deduplicated Kaggle run (~796K rows) through the documented pipeline.

Uses phishguard.pipelines.kaggle(full_dataset=True) unchanged, except for one runtime argument
the function already exposes: checkpoint_every=10**9. The default checkpoint rewrites the whole
enrichment CSV every 400 rows, which is what makes --full take "many hours". Output features are
identical; only the number of intermediate writes changes.

Canonicalization in this pipeline adds http:// to bare hosts, so domain grouping works (unlike the
retrain_with_fresh path behind the shipped models, see step 5b).
Artifacts: metrics/_work/full/ (gitignored).
"""

from __future__ import annotations

import importlib.util
import json
import os
import time

from common import RAW_KAGGLE_CSV, REPO, SEED, WORK, seed_everything, write_result

FULL = WORK / "full"
os.environ["PHISH_DATA_DIR"] = str(FULL / "data")
os.environ["PHISH_OUTPUTS_DIR"] = str(FULL / "outputs")
os.environ["PHISH_LOGS_DIR"] = str(FULL / "logs")
from common import isolate_env  # noqa: E402

isolate_env()
seed_everything()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phishguard import train as T  # noqa: E402
from phishguard.data.clean import canonicalize_url  # noqa: E402
from phishguard.features.layer1 import extract_layer1_features  # noqa: E402
from phishguard.logging_util import setup_logging  # noqa: E402
from phishguard.pipelines.kaggle import run_kaggle_pipeline  # noqa: E402
from phishguard.urls.safe import leak_safe_group_key  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", REPO / "metrics" / "scripts" / "02_evaluate_layer1.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def app_frame(urls, cols, num_cols):
    X = pd.DataFrame([extract_layer1_features(canonicalize_url(u)[0], use_dns=False) for u in urls])
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def main() -> None:
    setup_logging(FULL / "logs" / "full_run.log")
    t0 = time.perf_counter()
    run_kaggle_pipeline(
        csv_path=RAW_KAGGLE_CSV, seed=SEED, full_dataset=True, use_fresh_data=False,
        primary_selection="composite", write_primary_artifact=True, enrich_resume=False,
        checkpoint_every=10**9,
    )
    runtime = time.perf_counter() - t0
    out = FULL / "outputs"
    split = json.loads((out / "reports" / "split_leak_safe_stats.json").read_text())
    repo_metrics = json.loads((out / "metrics" / "metrics.json").read_text())

    proc = FULL / "data" / "processed"
    tr = pd.read_csv(proc / "kaggle_train.csv", dtype=str, low_memory=False)
    te = pd.read_csv(proc / "kaggle_test.csv", dtype=str, low_memory=False)
    tr["_split"], te["_split"] = "train", "test"
    full = pd.concat([tr, te], ignore_index=True)
    full = full.loc[full["url_length"].notna()].reset_index(drop=True)
    X, y, num_cols, _ = T._feature_matrix(full, T._exclude_layer1_only(full, True, include_dns=False))
    is_te = (full["_split"] == "test").values
    Xt, yt = X.loc[is_te].reset_index(drop=True), y[is_te]
    del tr, te

    held = {"dummy_most_frequent": ev.point_metrics(yt, np.zeros_like(yt), None)}
    pipes = {}
    for r in repo_metrics:
        m = r["model"]
        pipe = joblib.load(out / "models" / f"{m}.joblib")
        pipes[m] = pipe
        proba = ev.phish_proba(pipe, Xt)
        pred = (proba >= 0.5).astype(int)
        held[m] = {**ev.point_metrics(yt, pred, proba), "bootstrap_95ci": ev.bootstrap_ci(yt, pred, proba, n=1000),
                   "audit_official_https_mean_phish_proba": r.get("audit_official_https_mean_phish_proba")}
        print("scored", m, flush=True)
    winner = T._pick_layer1_primary_model_by_policy(repo_metrics, policy="composite")

    cols = list(X.columns)
    rec = pd.read_csv(REPO / "phishing_dataset" / "dataset_test_recent.csv")
    off_urls = rec.loc[rec["source"] == "brand_official", "url"].tolist()
    fresh = pd.read_csv(REPO / "phishing_dataset" / "dataset_full.csv")
    ph_urls = fresh.loc[fresh["source"] == "phishstats", "url"].tolist()
    Xo, Xp = app_frame(off_urls, cols, num_cols), app_frame(ph_urls, cols, num_cols)
    gtr = set(full.loc[~is_te, "canonical_url"].fillna("").map(lambda u: leak_safe_group_key(u)[0]))
    ext = {}
    for m, pipe in pipes.items():
        po, pp = ev.phish_proba(pipe, Xo), ev.phish_proba(pipe, Xp)
        ext[m] = {
            "official_brand_fp_rate_raw_0_5": float(np.mean(po >= 0.5)),
            "official_brand_fp_count": int(np.sum(po >= 0.5)),
            "phishstats_recall_raw_0_5": float(np.mean(pp >= 0.5)),
        }
    write_result(
        "06_full_run",
        {
            "command": "python metrics/scripts/06_full_run.py",
            "end_to_end_runtime_seconds": round(runtime, 1),
            "note": "run_kaggle_pipeline(full_dataset=True, checkpoint_every=10**9); Kaggle only, seed 42",
            "split": split,
            "n_features": len(cols),
            "held_out_test_threshold_0_5_raw": held,
            "winner_composite": winner,
            "winner_by_f1": T._pick_layer1_primary_model_by_policy(repo_metrics, policy="f1"),
            "external_sets_app_features": {
                "official_brand_urls_n": len(off_urls),
                "official_brand_domains_in_train": int(sum(leak_safe_group_key(canonicalize_url(u)[0])[0] in gtr for u in off_urls)),
                "phishstats_n": len(ph_urls),
                "phishstats_domains_in_train": int(sum(leak_safe_group_key(canonicalize_url(u)[0])[0] in gtr for u in ph_urls)),
                "per_model": ext,
            },
        },
    )


if __name__ == "__main__":
    main()
