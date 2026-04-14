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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline.label_policy import phish_probability_from_proba_row
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

# Scheme is a weak legitimacy signal in our Kaggle mix (many benign rows are http://; phish uses https).
LAYER1_EXCLUDE_FROM_X: Set[str] = {"has_https"}

# Used only for **choosing** ``layer1_primary.joblib`` among fitted models (documented; not a runtime allowlist).
LAYER1_OFFICIAL_HTTPS_AUDIT_URLS: Tuple[str, ...] = (
    "https://www.google.com/",
    "https://google.com",
    "https://www.amazon.com/",
    "https://amazon.com",
    "https://www.wikipedia.org/",
    "https://accounts.google.com",
    "https://paypal.com",
    "https://login.microsoftonline.com",
)


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
    "group_key_fallback_used",
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
    blocked = (set(df.columns) - allowed) | _exclude_for_training(exclude_fetch_proxy) | LAYER1_EXCLUDE_FROM_X
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


def _xgb_monotone_decreasing_trust(num_cols: List[str]) -> Tuple[int, ...]:
    """Trust features ↓ P(phish); ``https_without_official_anchor`` ↑ P(phish) (dataset artifact correction)."""
    trust_decrease = {
        "official_registrable_anchor",
        "official_domain_family",
        "brand_hostname_exact_label_match",
        "layer1_brand_trust_score",
        "official_anchor_with_https",
        "legit_auth_surface_on_official_anchor",
        "legit_admin_like_path_on_official_anchor",
        "legit_checkout_like_path_on_official_anchor",
        "path_shallow_le1",
        "path_shallow_le2",
        "no_authish_path_query_tokens",
        "simple_public_web_shape",
        "simple_official_homepage_shape",
    }
    https_mismatch_increase = {"https_without_official_anchor"}
    phish_risk_increase = {
        "suspicious_redirect_query_flag",
        "path_query_authish_keyword_hits",
        "path_segment_count",
    }
    return tuple(
        -1
        if c in trust_decrease
        else 1
        if c in https_mismatch_increase or c in phish_risk_increase
        else 0
        for c in num_cols
    )


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


def _mean_phish_proba_official_anchor_rows(pipe: Pipeline, X_te: pd.DataFrame) -> Any:
    """Mean P(phish) on test rows with ``official_registrable_anchor==1`` (realistic corp hosts in holdout)."""
    if "official_registrable_anchor" not in X_te.columns:
        return None
    mask = pd.to_numeric(X_te["official_registrable_anchor"], errors="coerce").fillna(0).astype(int) == 1
    if not bool(mask.any()):
        return None
    try:
        X_sub = X_te.loc[mask]
        P = pipe.predict_proba(X_sub)
        cls = np.asarray(pipe.classes_)
        ph = np.array([phish_probability_from_proba_row(P[i], cls) for i in range(len(P))])
        return float(np.mean(ph))
    except Exception:
        return None


def _audit_official_https_mean_phish(pipe: Pipeline, feature_columns: List[str]) -> float:
    """Mean raw P(phish) on a few canonical official HTTPS URLs (OOD vs Kaggle http-heavy benign rows)."""
    from src.app_v1.ml_layer1 import build_layer1_frame

    probs: List[float] = []
    for url in LAYER1_OFFICIAL_HTTPS_AUDIT_URLS:
        X_raw, _, _ = build_layer1_frame(url, use_dns=False)
        for c in feature_columns:
            if c not in X_raw.columns:
                X_raw[c] = np.nan
        X = X_raw[feature_columns]
        P = pipe.predict_proba(X)[0]
        cls = np.asarray(pipe.classes_)
        probs.append(phish_probability_from_proba_row(P, cls))
    return float(np.mean(probs))


def _pick_layer1_primary_model(metrics_all: List[Dict[str, Any]]) -> str:
    """Maximize F1 minus audit-URL mean phish (balances accuracy vs embarrassing false positives on big brands)."""
    if not metrics_all:
        raise ValueError("metrics_all empty")

    def composite(m: Dict[str, Any]) -> float:
        f1 = m.get("f1") or 0.0
        audit = m.get("audit_official_https_mean_phish_proba")
        return f1 if audit is None else f1 - float(audit)

    return str(max(metrics_all, key=composite)["model"])


def _evaluate(name: str, pipe: Pipeline, X_te: pd.DataFrame, y_te: np.ndarray) -> Dict[str, Any]:
    pred = pipe.predict(X_te)
    proba = None
    try:
        P = pipe.predict_proba(X_te)
        cls = np.asarray(pipe.classes_)
        proba = np.array([phish_probability_from_proba_row(P[i], cls) for i in range(len(P))])
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
    out["test_mean_phish_proba_official_anchor_rows"] = _mean_phish_proba_official_anchor_rows(pipe, X_te)
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
        P = pipe.predict_proba(X_te)
        cls = np.asarray(pipe.classes_)
        proba = np.array([phish_probability_from_proba_row(P[i], cls) for i in range(len(P))])
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

        xgb_kw: Dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1,
        }
        if layer1_only and not cat_cols:
            xgb_kw["monotone_constraints"] = _xgb_monotone_decreasing_trust(num_cols)
        models["xgboost"] = XGBClassifier(**xgb_kw)
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
        "label_semantics": {
            "legitimate_class": 0,
            "phishing_class": 1,
            "kaggle_raw": "1=legitimate, 0=phishing → mapped via kaggle_status_to_internal",
            "inference": "Use Pipeline.classes_ with phishing_class for predict_proba column; do not assume column order.",
        },
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
        "layer1_primary_selection": (
            "Argmax over fitted models of (f1 - audit_official_https_mean_phish_proba); "
            "audit_* is mean raw P(phish) on layer1_official_https_audit_urls only (model choice, not runtime rules)."
            if layer1_only
            else None
        ),
        "layer1_official_https_audit_urls": list(LAYER1_OFFICIAL_HTTPS_AUDIT_URLS) if layer1_only else None,
        "layer1_dropped_from_training_features": sorted(LAYER1_EXCLUDE_FROM_X) if layer1_only else None,
    }
    (reports_dir() / "training_config.json").write_text(json.dumps(training_meta, indent=2), encoding="utf-8")

    for name, est in models.items():
        pipe = _make_pipeline(est, num_cols, cat_cols)
        logger.info("Fitting %s …", name)
        pipe.fit(X_tr, y_tr)
        m = _evaluate(name, pipe, X_test, y_test)
        try:
            m["audit_official_https_mean_phish_proba"] = _audit_official_https_mean_phish(pipe, list(X.columns))
        except Exception as ex:
            logger.warning("Audit-URL mean phish skipped: %s", ex)
            m["audit_official_https_mean_phish_proba"] = None
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
    if layer1_only:
        try:
            primary_name = _pick_layer1_primary_model(metrics_all)
            best = next(m for m in metrics_all if m["model"] == primary_name)
        except (ValueError, StopIteration):
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
        best_name = _pick_layer1_primary_model(metrics_all)
        src = models_dir() / f"{best_name}.joblib"
        dst = models_dir() / "layer1_primary.joblib"
        if src.is_file():
            dst.write_bytes(src.read_bytes())
            row = next((m for m in metrics_all if m["model"] == best_name), {})
            logger.info(
                "Layer-1 primary -> %s (F1=%s, official_anchor_rows_mean_phish=%s)",
                best_name,
                row.get("f1"),
                row.get("test_mean_phish_proba_official_anchor_rows"),
            )
        _maybe_fit_layer1_probability_calibrator(X_val, y_val, X_test, y_test)
    return metrics_path


def _maybe_fit_layer1_probability_calibrator(
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> None:
    """Fit isotonic (fallback: Platt LR) on validation raw P(phish); persist for inference."""
    primary_path = models_dir() / "layer1_primary.joblib"
    if not primary_path.is_file():
        return
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression as LRPlatt

        pipe = joblib.load(primary_path)
        Pv = pipe.predict_proba(X_val)
        cls = np.asarray(pipe.classes_)
        raw_val = np.array([phish_probability_from_proba_row(Pv[i], cls) for i in range(len(Pv))])
        ctype = "isotonic"
        cal_model: Any = None
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_val, y_val)
            _ = iso.predict(raw_val)
            cal_model = iso
        except Exception as ex_iso:
            logger.warning("Isotonic calibration skipped (%s); using Platt LR on val scores.", ex_iso)
            lr = LRPlatt(max_iter=500, class_weight="balanced")
            lr.fit(raw_val.reshape(-1, 1), y_val)
            cal_model = lr
            ctype = "platt"

        Pt = pipe.predict_proba(X_test)
        raw_te = np.array([phish_probability_from_proba_row(Pt[i], cls) for i in range(len(Pt))])
        if ctype == "isotonic":
            cal_te = np.clip(cal_model.predict(raw_te), 0.0, 1.0)
        else:
            cal_te = np.clip(cal_model.predict_proba(raw_te.reshape(-1, 1))[:, 1], 0.0, 1.0)

        cal_path = models_dir() / "layer1_probability_calibrator.joblib"
        joblib.dump({"type": ctype, "model": cal_model, "fitted_on": "train_val_split_holdout"}, cal_path)

        b_raw = float(brier_score_loss(y_test, raw_te))
        b_cal = float(brier_score_loss(y_test, cal_te))
        rep = {
            "calibrator_type": ctype,
            "path": str(cal_path),
            "brier_test_raw_layer1_primary": b_raw,
            "brier_test_calibrated": b_cal,
        }
        (reports_dir() / "layer1_calibration_report.json").write_text(
            json.dumps(rep, indent=2),
            encoding="utf-8",
        )
        cfgp = reports_dir() / "training_config.json"
        if cfgp.is_file():
            tc = json.loads(cfgp.read_text(encoding="utf-8"))
            tc["layer1_probability_calibrator"] = cal_path.name
            tc["layer1_calibration_report"] = rep
            cfgp.write_text(json.dumps(tc, indent=2), encoding="utf-8")
        logger.info("Layer-1 probability calibration -> %s (Brier raw=%.4f cal=%.4f)", ctype, b_raw, b_cal)
    except Exception as ex:
        logger.warning("Layer-1 probability calibration failed: %s", ex)


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
