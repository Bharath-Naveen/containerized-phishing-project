"""Step 2: extended held-out evaluation of the 4 Layer-1 models from step 1, plus a dummy baseline.

Uses the exact fitted pipelines saved by step 1 (metrics/_work/outputs/models/*.joblib) and the
exact domain-grouped train/test split step 1 wrote. Adds what train.py does not report:
PR-AUC, false-positive rate, a DummyClassifier reference, 1,000-sample bootstrap 95% CIs,
5-fold StratifiedGroupKFold CV (mean +/- std), and the composite-rule winner.
"""

from __future__ import annotations

import json

from common import SEED, WORK, isolate_env, seed_everything, write_result

isolate_env()
seed_everything()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split  # noqa: E402

from phishguard import train as T  # noqa: E402
from phishguard.data.labels import phish_probability_from_proba_row  # noqa: E402
from phishguard.urls.safe import leak_safe_group_key  # noqa: E402

MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
N_BOOT = 1000


def load_split():
    proc = WORK / "data" / "processed"
    tr = pd.read_csv(proc / "kaggle_train.csv", dtype=str, low_memory=False)
    te = pd.read_csv(proc / "kaggle_test.csv", dtype=str, low_memory=False)
    tr["_split"] = "train"
    te["_split"] = "test"
    full = pd.concat([tr, te], ignore_index=True)
    m = full["url_length"].notna() & (full["url_length"].astype(str).str.len() > 0)
    full = full.loc[m].reset_index(drop=True)
    exclude = T._exclude_layer1_only(full, True, include_dns=False)
    X, y, num_cols, cat_cols = T._feature_matrix(full, exclude)
    is_train = (full["_split"] == "train").values
    groups = np.array([leak_safe_group_key(u)[0] for u in full["canonical_url"].fillna("").astype(str)])
    return full, X, y, is_train, groups


def phish_proba(pipe, X):
    P = pipe.predict_proba(X)
    cls = np.asarray(pipe.classes_)
    return np.array([phish_probability_from_proba_row(P[i], cls) for i in range(len(P))])


def point_metrics(y, pred, proba):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "accuracy": float((tp + tn) / len(y)),
        "confusion_matrix_[[tn,fp],[fn,tp]]": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if proba is not None and len(np.unique(proba)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    else:
        out["roc_auc"] = 0.5
        out["pr_auc"] = float(np.mean(y))
    return out


def bootstrap_ci(y, pred, proba, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    keys = ["precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate"]
    acc = {k: [] for k in keys}
    idx_all = np.arange(len(y))
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(y), replace=True)
        yb, pb = y[idx], pred[idx]
        acc["precision"].append(precision_score(yb, pb, zero_division=0))
        acc["recall"].append(recall_score(yb, pb, zero_division=0))
        acc["f1"].append(f1_score(yb, pb, zero_division=0))
        neg = yb == 0
        acc["false_positive_rate"].append(float(np.mean(pb[neg] == 1)) if neg.any() else 0.0)
        if proba is not None and len(np.unique(proba[idx])) > 1:
            acc["roc_auc"].append(roc_auc_score(yb, proba[idx]))
            acc["pr_auc"].append(average_precision_score(yb, proba[idx]))
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items() if v
    }


def main() -> None:
    full, X, y, is_train, groups = load_split()
    X_train, y_train = X.loc[is_train].reset_index(drop=True), y[is_train]
    X_test, y_test = X.loc[~is_train].reset_index(drop=True), y[~is_train]
    g_train, g_test = set(groups[is_train]), set(groups[~is_train])
    # Same inner split train.py uses (val_size=0.15, stratified, seed 42).
    X_tr, _X_val, y_tr, _y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    mdir = WORK / "outputs" / "models"
    pipes = {m: joblib.load(mdir / f"{m}.joblib") for m in MODELS}
    repo_metrics = json.loads((WORK / "outputs" / "metrics" / "metrics.json").read_text())
    audit = {r["model"]: r.get("audit_official_https_mean_phish_proba") for r in repo_metrics}

    held_out = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    d_pred = dummy.predict(X_test)
    held_out["dummy_most_frequent"] = {
        **point_metrics(y_test, d_pred, None),
        "note": "Always predicts the majority class of the training rows (legitimate).",
    }
    dummy_s = DummyClassifier(strategy="stratified", random_state=SEED).fit(X_tr, y_tr)
    ds_pred = dummy_s.predict(X_test)
    held_out["dummy_stratified"] = {
        **point_metrics(y_test, ds_pred, dummy_s.predict_proba(X_test)[:, 1]),
        "note": "Random guesses in training class proportions.",
    }
    for m, pipe in pipes.items():
        proba = phish_proba(pipe, X_test)
        pred = pipe.predict(X_test)
        held_out[m] = {
            **point_metrics(y_test, pred, proba),
            "bootstrap_95ci": bootstrap_ci(y_test, pred, proba),
            "audit_official_https_mean_phish_proba": audit.get(m),
        }
        held_out[m]["composite_score_f1_minus_audit"] = held_out[m]["f1"] - float(audit.get(m) or 0.0)

    winner = T._pick_layer1_primary_model_by_policy(repo_metrics, policy="composite")
    by_f1 = T._pick_layer1_primary_model_by_policy(repo_metrics, policy="f1")
    by_auc = T._pick_layer1_primary_model_by_policy(repo_metrics, policy="roc_auc")

    # App threshold view for the composite winner: calibrated P(phish) >= 0.5 is the Layer-1 flag
    # (ml_layer1.predict_layer1), and ML-only 3-way verdict uses combined = 0.65 * calibrated
    # (org risk 0 without capture) vs verdict_policy thresholds 0.56 / 0.38.
    cal = joblib.load(mdir / "layer1_probability_calibrator.joblib")
    raw_w = phish_proba(pipes[winner], X_test)
    cal_w = np.clip(cal["model"].predict(raw_w), 0, 1) if cal["type"] == "isotonic" else raw_w
    pred_cal = (cal_w >= 0.5).astype(int)
    combined = 0.65 * cal_w
    v_phish = combined >= 0.56
    v_legit = combined <= 0.38
    app_view = {
        "model": winner,
        "layer1_flag_calibrated_ge_0_5": point_metrics(y_test, pred_cal, cal_w),
        "ml_only_3way_verdict_counts": {
            "phishing_rows": {
                "likely_phishing": int(np.sum(v_phish & (y_test == 1))),
                "uncertain": int(np.sum(~v_phish & ~v_legit & (y_test == 1))),
                "likely_legitimate": int(np.sum(v_legit & (y_test == 1))),
            },
            "legit_rows": {
                "likely_phishing": int(np.sum(v_phish & (y_test == 0))),
                "uncertain": int(np.sum(~v_phish & ~v_legit & (y_test == 0))),
                "likely_legitimate": int(np.sum(v_legit & (y_test == 0))),
            },
        },
        "note": "ML-only proxy (no Playwright capture). combined=0.65*calibrated as in benchmark_url_suites.py.",
    }

    # 5-fold StratifiedGroupKFold CV on all rows (train+test), refitting clones of each pipeline.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = {m: [] for m in ["dummy_most_frequent"] + MODELS}
    fold_overlap = []
    for k, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_overlap.append(len(set(groups[tr_idx]) & set(groups[te_idx])))
        Xa, ya, Xb, yb = X.iloc[tr_idx], y[tr_idx], X.iloc[te_idx], y[te_idx]
        dm = DummyClassifier(strategy="most_frequent").fit(Xa, ya)
        cv["dummy_most_frequent"].append(point_metrics(yb, dm.predict(Xb), None))
        for m, pipe in pipes.items():
            p = clone(pipe).fit(Xa, ya)
            cv[m].append(point_metrics(yb, p.predict(Xb), phish_proba(p, Xb)))
        print(f"cv fold {k + 1}/5 done")
    cv_summary = {}
    for m, folds in cv.items():
        cv_summary[m] = {
            key: {"mean": float(np.mean([f[key] for f in folds])), "std": float(np.std([f[key] for f in folds], ddof=1))}
            for key in ["precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate"]
        }

    write_result(
        "02_layer1_eval",
        {
            "command": "python metrics/scripts/02_evaluate_layer1.py",
            "data": {
                "source": "Kaggle harisudhan411/phishing-and-legitimate-urls, 50,000-row stratified sample (seed 42) + 896 curated legit URLs added by the repo's simple_legit_augment",
                "n_rows_total": int(len(y)),
                "n_train_rows": int(is_train.sum()),
                "n_train_rows_used_for_fit_after_15pct_val_holdout": int(len(y_tr)),
                "n_test_rows": int((~is_train).sum()),
                "test_class_counts": {"legit_0": int(np.sum(y_test == 0)), "phish_1": int(np.sum(y_test == 1))},
                "train_groups": len(g_train),
                "test_groups": len(g_test),
                "registered_domain_overlap_train_test": len(g_train & g_test),
                "n_features": int(X.shape[1]),
            },
            "decision_threshold": "0.5 on raw model P(phish) (sklearn predict), same as src/pipeline/train.py",
            "held_out_test": held_out,
            "composite_rule": "argmax(F1 - mean raw P(phish) on 8 official HTTPS URLs) (src/pipeline/train.py)",
            "winner_composite": winner,
            "winner_by_f1": by_f1,
            "winner_by_roc_auc": by_auc,
            "app_threshold_view_for_winner": app_view,
            "cv_5fold_stratified_group": cv_summary,
            "cv_group_overlap_per_fold": fold_overlap,
            "bootstrap": f"{N_BOOT} row-level resamples of the held-out test set, percentile 95% CI",
        },
    )


if __name__ == "__main__":
    main()
