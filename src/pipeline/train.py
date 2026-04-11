"""Train sklearn pipelines (logistic regression, random forest, optional boosting)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline.layer1_features import layer1_feature_key_set
from src.pipeline.paths import ensure_layout, figures_dir, metrics_dir, models_dir, processed_dir, reports_dir

logger = logging.getLogger(__name__)

# Proxy features that often encode “fetch worked” vs “dead domain” more than page semantics.
FETCH_PROXY_FEATURES: Set[str] = {
    "page_fetch_success",
    "page_load_success",
    "fetch_features_missing",
    "fetch_error_flag",
    "http_status",
    "html_features_missing",
    "behavior_skipped",
    "dns_missing",
    "dns_error",
    "navigation_timeout",
    "automation_blocked",
    "challenge_detected",
}

BASE_EXCLUDE_FROM_X = {
    "url",
    "source_file",
    "source_dataset",
    "canonical_url",
    "final_url",
    "fetch_error",
    "parse_error",
    "enrich_fatal_error",
    "label",
    "source_brand_hint",
    "action_category",
    "kaggle_raw_status",
    "invalid_url",
    "_split",
}


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", max_categories=30, sparse=False)


def _exclude_for_training(exclude_fetch_proxy: bool) -> Set[str]:
    s = set(BASE_EXCLUDE_FROM_X)
    if exclude_fetch_proxy:
        s |= FETCH_PROXY_FEATURES
    return s


def _exclude_layer1_only(df: pd.DataFrame, exclude_fetch_proxy: bool, *, include_dns: bool) -> Set[str]:
    allowed = layer1_feature_key_set(include_dns=include_dns)
    blocked = (set(df.columns) - allowed) | _exclude_for_training(exclude_fetch_proxy)
    return blocked


def _feature_matrix(
    df: pd.DataFrame, exclude_cols: Set[str]
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    y = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int).values
    use_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[use_cols].copy()
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            X[c] = pd.to_numeric(s, errors="coerce")
            num_cols.append(c)
            continue
        coerced = pd.to_numeric(s.astype(str), errors="coerce")
        ratio = float(coerced.notna().sum()) / max(len(s), 1)
        if ratio > 0.85:
            X[c] = coerced
            num_cols.append(c)
        else:
            nu = s.fillna("missing").astype(str).nunique(dropna=False)
            if nu <= 40:
                X[c] = s.fillna("missing").astype(str)
                cat_cols.append(c)
            else:
                X.drop(columns=[c], inplace=True)
    if not num_cols and not cat_cols:
        raise ValueError("No feature columns after preprocessing.")
    return X, y, num_cols, cat_cols


def _make_pipeline(model: Any, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    transformers: List[Tuple[str, Any, List[str]]] = []
    if num_cols:
        num_tr = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("num", num_tr, num_cols))
    if cat_cols:
        cat_tr = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    _one_hot_encoder(),
                ),
            ]
        )
        transformers.append(("cat", cat_tr, cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("prep", pre), ("model", model)])


def _evaluate(name: str, pipe: Pipeline, X_te: pd.DataFrame, y_te: np.ndarray) -> Dict[str, Any]:
    pred = pipe.predict(X_te)
    proba = None
    try:
        proba = pipe.predict_proba(X_te)[:, 1]
    except Exception:
        proba = None
    out: Dict[str, Any] = {"model": name}
    out["accuracy"] = float(accuracy_score(y_te, pred))
    out["precision"] = float(precision_score(y_te, pred, zero_division=0))
    out["recall"] = float(recall_score(y_te, pred, zero_division=0))
    out["f1"] = float(f1_score(y_te, pred, zero_division=0))
    if proba is not None and len(np.unique(y_te)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_te, proba))
        except Exception:
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None
    out["confusion_matrix"] = confusion_matrix(y_te, pred).tolist()
    return out


def _export_misclassifications(
    name: str,
    pipe: Pipeline,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    meta_te: pd.DataFrame,
) -> None:
    pred = pipe.predict(X_te)
    proba = None
    try:
        proba = pipe.predict_proba(X_te)[:, 1]
    except Exception:
        pass
    wrong = pred != y_te
    if not wrong.any():
        path = reports_dir() / f"misclassifications_{name}.csv"
        path.write_text("canonical_url,url,y_true,y_pred,phish_proba,note\n", encoding="utf-8")
        return
    rows = meta_te.loc[wrong].copy()
    rows["y_true"] = y_te[wrong]
    rows["y_pred"] = pred[wrong]
    if proba is not None:
        rows["phish_proba"] = proba[wrong]
    else:
        rows["phish_proba"] = np.nan
    keep = [c for c in ("canonical_url", "url", "y_true", "y_pred", "phish_proba") if c in rows.columns]
    rows[keep].to_csv(reports_dir() / f"misclassifications_{name}.csv", index=False)


def train(
    train_csv: Path | None = None,
    test_csv: Path | None = None,
    *,
    val_size: float = 0.15,
    random_state: int = 42,
    exclude_fetch_proxy_features: bool = True,
    layer1_only: bool = False,
    layer1_include_dns: bool = False,
) -> Path:
    ensure_layout()
    tr_path = train_csv or (processed_dir() / "train.csv")
    te_path = test_csv or (processed_dir() / "test.csv")
    train_df = pd.read_csv(tr_path, dtype=str, low_memory=False)
    test_df = pd.read_csv(te_path, dtype=str, low_memory=False)
    if train_df.empty or test_df.empty:
        logger.error("Train or test empty; abort training.")
        raise SystemExit(1)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["_split"] = "train"
    test_df["_split"] = "test"
    full = pd.concat([train_df, test_df], ignore_index=True)

    feat_check_col = "url_length" if "url_length" in full.columns else None
    if feat_check_col:
        m = full[feat_check_col].notna() & (full[feat_check_col].astype(str).str.len() > 0)
        full = full.loc[m].copy()
    full = full.reset_index(drop=True)
    if len(full) < 10:
        logger.error("Too few rows with feature columns after filtering; abort.")
        raise SystemExit(1)

    is_train = full["_split"].astype(str) == "train"
    meta_cols = [c for c in ("canonical_url", "url", "label") if c in full.columns]
    meta_full = full[meta_cols].copy() if meta_cols else pd.DataFrame(index=full.index)

    if layer1_only:
        exclude = _exclude_layer1_only(
            full, exclude_fetch_proxy_features, include_dns=layer1_include_dns
        )
    else:
        exclude = _exclude_for_training(exclude_fetch_proxy_features)
    X, y, num_cols, cat_cols = _feature_matrix(full, exclude)
    if "action_category" in X.columns:
        logger.error("action_category must not be used as a model feature; check BASE_EXCLUDE_FROM_X.")
        raise SystemExit(2)

    X_train = X.loc[is_train].reset_index(drop=True)
    X_test = X.loc[~is_train].reset_index(drop=True)
    y_train = y[is_train.values]
    y_test = y[~is_train.values]
    meta_test = meta_full.loc[~is_train.values].reset_index(drop=True)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train if len(np.unique(y_train)) > 1 else None,
    )

    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=400, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        logger.info("XGBoost not installed; skipping.")
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        logger.info("LightGBM not installed; skipping.")

    metrics_all: List[Dict[str, Any]] = []
    training_meta = {
        "exclude_fetch_proxy_features": exclude_fetch_proxy_features,
        "layer1_only": layer1_only,
        "layer1_include_dns": layer1_include_dns if layer1_only else None,
        "n_train_after_filter": int(is_train.sum()),
        "n_test_after_filter": int((~is_train).sum()),
        "dropped_fetch_proxy_columns": sorted(FETCH_PROXY_FEATURES) if exclude_fetch_proxy_features else [],
        "action_category_excluded_from_features": True,
        "action_category_present_in_rows": "action_category" in full.columns,
        "metadata_not_used_as_features": [
            c
            for c in (
                "action_category",
                "source_brand_hint",
                "source_dataset",
                "source_file",
                "kaggle_raw_status",
            )
            if c in full.columns
        ],
        "feature_columns_used": list(X.columns),
    }
    (reports_dir() / "training_config.json").write_text(json.dumps(training_meta, indent=2), encoding="utf-8")

    for name, est in models.items():
        pipe = _make_pipeline(est, num_cols, cat_cols)
        logger.info("Fitting %s …", name)
        pipe.fit(X_tr, y_tr)
        m = _evaluate(name, pipe, X_test, y_test)
        metrics_all.append(m)
        joblib.dump(pipe, models_dir() / f"{name}.joblib")

        _export_misclassifications(name, pipe, X_test, y_test, meta_test)

        try:
            model_step = pipe.named_steps["model"]
            prep = pipe.named_steps["prep"]
            feat_names = prep.get_feature_names_out()
            if hasattr(model_step, "feature_importances_"):
                imp = model_step.feature_importances_
                top = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:50]
                (reports_dir() / f"importance_{name}.csv").write_text(
                    "feature,importance\n" + "\n".join(f"{a},{float(b)}" for a, b in top),
                    encoding="utf-8",
                )
            if hasattr(model_step, "coef_"):
                coef = model_step.coef_.ravel()
                (reports_dir() / f"coef_{name}.csv").write_text(
                    "index,coefficient\n" + "\n".join(f"{i},{float(c)}" for i, c in enumerate(coef)),
                    encoding="utf-8",
                )
        except Exception as ex:
            logger.debug("Importance export skipped: %s", ex)

    metrics_path = metrics_dir() / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_all, indent=2), encoding="utf-8")
    pd.DataFrame(metrics_all).to_csv(metrics_dir() / "metrics_summary.csv", index=False)

    best = max(metrics_all, key=lambda m: m.get("f1") or 0)
    try:
        import matplotlib.pyplot as plt

        name = best["model"]
        pipe = joblib.load(models_dir() / f"{name}.joblib")
        pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(cm)
        ax.set_title(f"Confusion — {name}")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "k")
        ax.set_ylabel("True")
        ax.set_xlabel("Pred")
        fig.tight_layout()
        fig.savefig(figures_dir() / f"confusion_{name}.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        logger.debug("Figure export skipped: %s", e)

    logger.info("Training complete. Metrics -> %s", metrics_path)
    if layer1_only and metrics_all:
        best_name = max(metrics_all, key=lambda m: m.get("f1") or 0)["model"]
        src = models_dir() / f"{best_name}.joblib"
        dst = models_dir() / "layer1_primary.joblib"
        if src.is_file():
            dst.write_bytes(src.read_bytes())
            logger.info("Copied best Layer-1 model %s -> %s", best_name, dst)
    return metrics_path


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "train.log")
    p = argparse.ArgumentParser(description="Train ML models on enriched features.")
    p.add_argument("--train", type=Path, default=None)
    p.add_argument("--test", type=Path, default=None)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument(
        "--include-fetch-proxy-features",
        action="store_true",
        help="Keep page_fetch_success / http_status etc. as features (more leakage risk).",
    )
    p.add_argument(
        "--layer1-only",
        action="store_true",
        help="Train only on fast URL/hosting/(optional DNS) columns from layer1 enrichment.",
    )
    p.add_argument(
        "--layer1-include-dns",
        action="store_true",
        help="With --layer1-only, allow DNS columns (must match enrichment).",
    )
    args = p.parse_args()
    train(
        args.train,
        args.test,
        val_size=args.val_size,
        exclude_fetch_proxy_features=not args.include_fetch_proxy_features,
        layer1_only=args.layer1_only,
        layer1_include_dns=args.layer1_include_dns,
    )


if __name__ == "__main__":
    main()
