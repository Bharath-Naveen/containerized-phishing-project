"""Step 5b (methodology fix, evaluation only): the shipped models' split and features, re-checked.

Finding: src/pipeline/retrain_with_fresh.py sets canonical_url = raw url when the column is missing,
so Kaggle rows without a scheme (for example "summah.info/") reach feature extraction and grouping
unparsed. Two consequences for run 20260427_234728 (the shipped models):
  1. Grouping: leak_safe_group_key() returns "malformed::<hash>" (one group per URL), so the
     "domain-grouped" split did not group by domain. Many test rows share a real registered domain
     with training rows.
  2. Train/serve skew: host features (hostname_length, num_dots, entropy, ...) are 0 / missing for
     most Kaggle rows at training time, but the app canonicalizes to http://... at inference, so the
     same URL gets different features in production.

This script does not retrain or modify anything. It re-scores the shipped models:
  (a) as reported:   all test rows, features from the saved CSV (what train.py measured)
  (b) leak-free:     only test rows whose real registered domain never appears in train
  (c) leak-free, app features: (b) with features recomputed the way ml_layer1.predict_layer1 does
"""

from __future__ import annotations

import importlib.util

from common import REPO, isolate_env, seed_everything, write_result

isolate_env()
seed_everything()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.clean import canonicalize_url  # noqa: E402
from src.pipeline.layer1_features import extract_layer1_features  # noqa: E402
from src.pipeline.safe_url import leak_safe_group_key  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", REPO / "metrics" / "scripts" / "02_evaluate_layer1.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

RUN = "20260427_234728"
PROC = REPO / "data" / "processed"
MD = REPO / "outputs" / "models"
MODELS = {
    "random_forest (deployed layer1_primary)": "layer1_primary.joblib",
    "logistic_regression (witness)": "logistic_regression.joblib",
    "xgboost (witness)": "xgboost.joblib",
    "lightgbm (witness)": "lightgbm.joblib",
}


def true_group(u: str) -> str:
    return leak_safe_group_key(canonicalize_url(str(u or ""))[0])[0]


def as_frame(df_feats: pd.DataFrame, cols) -> pd.DataFrame:
    X = df_feats.copy()
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[cols].apply(pd.to_numeric, errors="coerce")


def main() -> None:
    tr = pd.read_csv(PROC / f"retrain_with_fresh_train_{RUN}.csv", dtype=str, low_memory=False)
    te = pd.read_csv(PROC / f"retrain_with_fresh_test_{RUN}.csv", dtype=str, low_memory=False)
    y = te["label"].astype(int).values

    stored_key_tr = tr["canonical_url"].fillna("").map(lambda u: leak_safe_group_key(u)[0])
    n_malformed = int(stored_key_tr.str.startswith("malformed::").sum())
    gtr = set(tr["canonical_url"].fillna("").map(true_group))
    gte = te["canonical_url"].fillna("").map(true_group)
    clean = ~gte.isin(gtr).values

    hfm = {
        f"{src}|label={lab}": {"rows": int(len(g)), "hosting_features_missing_1": int((g["hosting_features_missing"] == "1").sum())}
        for (src, lab), g in tr.groupby([tr["source"].fillna("na"), tr["label"]])
    }

    pipes = {k: joblib.load(MD / v) for k, v in MODELS.items()}
    cols = list(pipes["random_forest (deployed layer1_primary)"].named_steps["prep"].feature_names_in_)
    X_csv = as_frame(te, cols)
    print("recomputing app-path features for", int(clean.sum()), "rows", flush=True)
    te_clean = te.loc[clean]
    X_app = as_frame(
        pd.DataFrame([extract_layer1_features(canonicalize_url(u)[0], use_dns=False) for u in te_clean["canonical_url"].fillna("")]),
        cols,
    )
    yc = y[clean]

    out = {}
    for name, pipe in pipes.items():
        pa = ev.phish_proba(pipe, X_csv)
        pc = pa[clean]
        pca = ev.phish_proba(pipe, X_app)
        out[name] = {
            "a_as_reported_all_test_csv_features": ev.point_metrics(y, (pa >= 0.5).astype(int), pa),
            "b_leak_free_rows_csv_features": ev.point_metrics(yc, (pc >= 0.5).astype(int), pc),
            "c_leak_free_rows_app_features": {
                **ev.point_metrics(yc, (pca >= 0.5).astype(int), pca),
                "bootstrap_95ci": ev.bootstrap_ci(yc, (pca >= 0.5).astype(int), pca),
            },
        }
        print("scored", name, flush=True)
    out["dummy_most_frequent_leak_free_rows"] = ev.point_metrics(yc, np.zeros_like(yc), None)

    write_result(
        "05b_deployed_leakfix",
        {
            "command": "python metrics/scripts/05b_deployed_model_leakfix_eval.py",
            "deployed_run": RUN,
            "train_rows_with_malformed_group_key": n_malformed,
            "train_rows": int(len(tr)),
            "hosting_features_missing_by_source_label_train": hfm,
            "test_rows": int(len(te)),
            "test_rows_sharing_real_domain_with_train": int((~clean).sum()),
            "test_rows_leak_free": int(clean.sum()),
            "leak_free_class_counts": {"legit_0": int(np.sum(yc == 0)), "phish_1": int(np.sum(yc == 1))},
            "threshold": "raw P(phish) >= 0.5",
            "results": out,
        },
    )


if __name__ == "__main__":
    main()
