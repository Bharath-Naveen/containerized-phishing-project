"""Step 5: verify the model the app actually ships (outputs/models/layer1_primary.joblib).

The deployed Random Forest and its three witness models come from fresh-retrain run 20260427_234728
(500,000-row stratified sample of Kaggle + 274 PhishStats rows, composite rule). Their byte hashes
match outputs/fresh_retrain_runs/20260427_234728/models/*.joblib. That run's train/test CSVs were
saved in data/processed/, so this step re-scores the deployed models on the saved held-out split
(no retraining) and checks:
  * held-out metrics + dummy baseline + bootstrap CI, domain overlap of the saved split
  * false positives on 298 curated official brand URLs (phishing_dataset/dataset_test_recent.csv,
    collected 2026-04-26, not used in training)
  * recall on the 284 PhishStats phishing rows, split into rows that were / were not in training
  * calibrator mismatch: deploy_layer1_primary.py swaps the model but keeps the older isotonic
    calibrator. A refit calibrator is written to metrics/_work (NOT to outputs/models).
"""

from __future__ import annotations

import importlib.util

from common import REPO, SEED, WORK, isolate_env, seed_everything, sha256, write_result

isolate_env()
seed_everything()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from phishguard import train as T  # noqa: E402
from phishguard.data.clean import canonicalize_url  # noqa: E402
from phishguard.features.layer1 import extract_layer1_features  # noqa: E402
from phishguard.urls.safe import leak_safe_group_key  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", REPO / "metrics" / "audit_baseline" / "scripts" / "02_evaluate_layer1.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

RUN = "20260427_234728"
PROC = REPO / "data" / "processed"
MD = REPO / "outputs" / "models"


def main() -> None:
    tr = pd.read_csv(PROC / f"retrain_with_fresh_train_{RUN}.csv", dtype=str, low_memory=False)
    te = pd.read_csv(PROC / f"retrain_with_fresh_test_{RUN}.csv", dtype=str, low_memory=False)
    tr["_split"], te["_split"] = "train", "test"
    full = pd.concat([tr, te], ignore_index=True)
    full = full.loc[full["url_length"].notna() & (full["url_length"].astype(str).str.len() > 0)].reset_index(drop=True)
    X, y, num_cols, cat_cols = T._feature_matrix(full, T._exclude_layer1_only(full, True, include_dns=False))
    is_tr = (full["_split"] == "train").values
    Xtr, ytr = X.loc[is_tr].reset_index(drop=True), y[is_tr]
    Xte, yte = X.loc[~is_tr].reset_index(drop=True), y[~is_tr]
    # Group by the REAL registered domain (canonicalize first; stored keys are broken for scheme-less rows, see 05b).
    g = np.array([leak_safe_group_key(canonicalize_url(u)[0])[0] for u in full["canonical_url"].fillna("").astype(str)])
    gtr, gte = set(g[is_tr]), set(g[~is_tr])
    url_tr = set(full.loc[is_tr, "canonical_url"].astype(str))

    models = {
        "random_forest (deployed layer1_primary)": MD / "layer1_primary.joblib",
        "logistic_regression (witness)": MD / "logistic_regression.joblib",
        "xgboost (witness)": MD / "xgboost.joblib",
        "lightgbm (witness)": MD / "lightgbm.joblib",
    }
    held = {}
    X_fit, X_val, y_fit, y_val = train_test_split(Xtr, ytr, test_size=0.15, random_state=SEED, stratify=ytr)
    dm = DummyClassifier(strategy="most_frequent").fit(X_fit, y_fit)
    held["dummy_most_frequent"] = ev.point_metrics(yte, dm.predict(Xte), None)
    pipes = {}
    for name, p in models.items():
        pipe = joblib.load(p)
        pipes[name] = pipe
        proba = ev.phish_proba(pipe, Xte)
        pred = pipe.predict(Xte)
        held[name] = {**ev.point_metrics(yte, pred, proba), "bootstrap_95ci": ev.bootstrap_ci(yte, pred, proba), "sha256": sha256(p)}
        print("scored", name, flush=True)

    rf = pipes["random_forest (deployed layer1_primary)"]
    cols = list(X.columns)

    # Calibration: shipped calibrator vs a calibrator refit on this model's own validation slice.
    shipped = joblib.load(MD / "layer1_probability_calibrator.joblib")
    raw_te = ev.phish_proba(rf, Xte)
    raw_val = ev.phish_proba(rf, X_val)
    shipped_te = np.clip(shipped["model"].predict(raw_te), 0, 1)
    refit = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val)
    refit_te = np.clip(refit.predict(raw_te), 0, 1)
    WORK.mkdir(parents=True, exist_ok=True)
    refit_path = WORK / "deployed_rf_refit_calibrator.joblib"
    joblib.dump({"type": "isotonic", "model": refit, "fitted_on": f"15% val slice of run {RUN} train split, seed 42"}, refit_path)
    calib = {
        "shipped_calibrator_sha256": sha256(MD / "layer1_probability_calibrator.joblib"),
        "note": "deploy_layer1_primary.py copies the model only; the calibrator on disk was fit for a different model (file dated before the RF was deployed).",
        "brier_raw": float(brier_score_loss(yte, raw_te)),
        "brier_shipped_calibrator": float(brier_score_loss(yte, shipped_te)),
        "brier_refit_calibrator": float(brier_score_loss(yte, refit_te)),
        "flag_at_calibrated_0_5_shipped": ev.point_metrics(yte, (shipped_te >= 0.5).astype(int), shipped_te),
        "flag_at_calibrated_0_5_refit": ev.point_metrics(yte, (refit_te >= 0.5).astype(int), refit_te),
        "refit_calibrator_written_to": str(refit_path.relative_to(REPO)),
    }

    # Official brand URLs (curated, collected 2026-04-26; not in training per training_data_context).
    rec = pd.read_csv(REPO / "phishing_dataset" / "dataset_test_recent.csv")
    off = rec[rec["source"] == "brand_official"]
    Xo = pd.DataFrame([extract_layer1_features(canonicalize_url(u)[0], use_dns=False) for u in off["url"]])
    for c in cols:
        if c not in Xo.columns:
            Xo[c] = np.nan
    Xo = Xo[cols]
    for c in num_cols:
        Xo[c] = pd.to_numeric(Xo[c], errors="coerce")
    po = ev.phish_proba(rf, Xo)
    po_ship = np.clip(shipped["model"].predict(po), 0, 1)
    po_refit = np.clip(refit.predict(po), 0, 1)
    off_dom = [leak_safe_group_key(canonicalize_url(u)[0])[0] for u in off["url"]]
    official = {
        "n_urls": int(len(off)),
        "n_urls_exactly_in_train": int(sum(canonicalize_url(u)[0] in url_tr for u in off["url"])),
        "n_urls_whose_domain_in_train": int(sum(d in gtr for d in off_dom)),
        "fp_rate_raw_0_5": float(np.mean(po >= 0.5)),
        "fp_rate_shipped_calibrated_0_5": float(np.mean(po_ship >= 0.5)),
        "fp_rate_refit_calibrated_0_5": float(np.mean(po_refit >= 0.5)),
        "mean_raw_p_phish": float(np.mean(po)),
        "fp_count_raw_0_5": int(np.sum(po >= 0.5)),
    }

    # PhishStats fresh phishing rows.
    full_fresh = pd.read_csv(REPO / "phishing_dataset" / "dataset_full.csv")
    ph = full_fresh[full_fresh["source"] == "phishstats"].copy()
    ph["canon"] = [canonicalize_url(u)[0] for u in ph["url"]]
    ph["in_train"] = ph["canon"].isin(url_tr)
    Xp = pd.DataFrame([extract_layer1_features(c, use_dns=False) for c in ph["canon"]])
    for c in cols:
        if c not in Xp.columns:
            Xp[c] = np.nan
    Xp = Xp[cols]
    for c in num_cols:
        Xp[c] = pd.to_numeric(Xp[c], errors="coerce")
    pp = ev.phish_proba(rf, Xp) >= 0.5
    fresh = {
        "n_phishstats_rows": int(len(ph)),
        "n_in_deployed_train_split": int(ph["in_train"].sum()),
        "recall_all_rows": float(pp.mean()),
        "recall_rows_in_train": float(pp[ph["in_train"].values].mean()) if ph["in_train"].any() else None,
        "recall_rows_not_in_train": float(pp[~ph["in_train"].values].mean()) if (~ph["in_train"]).any() else None,
        "n_rows_not_in_train": int((~ph["in_train"]).sum()),
        "n_rows_whose_domain_in_train": int(sum(leak_safe_group_key(c)[0] in gtr for c in ph["canon"])),
        "collection_dates": [str(ph["collection_date"].min()), str(ph["collection_date"].max())],
    }

    write_result(
        "05_deployed_model",
        {
            "command": "python metrics/scripts/05_deployed_model.py",
            "deployed_run": RUN,
            "data": {
                "train_csv_sha256": sha256(PROC / f"retrain_with_fresh_train_{RUN}.csv"),
                "test_csv_sha256": sha256(PROC / f"retrain_with_fresh_test_{RUN}.csv"),
                "n_train": int(is_tr.sum()),
                "n_test": int((~is_tr).sum()),
                "test_class_counts": {"legit_0": int(np.sum(yte == 0)), "phish_1": int(np.sum(yte == 1))},
                "registered_domains_shared_by_train_and_test": len(gtr & gte),
                "n_features": len(cols),
            },
            "held_out_test_threshold_0_5_raw": held,
            "calibration": calib,
            "official_brand_urls": official,
            "fresh_phishstats": fresh,
        },
    )


if __name__ == "__main__":
    main()
