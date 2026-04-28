"""Load Layer-1 sklearn pipeline and score a single URL (fast URL/host features only)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.pipeline.clean import canonicalize_url
from src.pipeline.features.brand_signals import (
    BRAND_STRUCTURE_FEATURE_KEYS,
    explain_brand_structure_features,
)
from src.pipeline.label_policy import INTERNAL_PHISH, phish_probability_from_proba_row
from src.pipeline.layer1_features import extract_layer1_features
from src.pipeline.paths import models_dir, reports_dir

logger = logging.getLogger(__name__)

# Optional witness models (same feature space as Layer-1 primary). Missing files are skipped.
_AGREEMENT_MODEL_FILES: Tuple[Tuple[str, str], ...] = (
    ("logistic_regression", "logistic_regression.joblib"),
    ("random_forest", "random_forest.joblib"),
    ("xgboost", "xgboost.joblib"),
    ("lightgbm", "lightgbm.joblib"),
)


def _quorum_strong_votes(n: int) -> int:
    """Minimum count on one side for strong_{phishing|legitimate} (ceil(0.75 * n))."""
    if n <= 0:
        return 0
    return int(math.ceil(0.75 * n))


def _raw_phish_proba_from_pipeline(pipe: Any, X: pd.DataFrame) -> Tuple[Optional[float], Optional[str]]:
    try:
        proba = np.asarray(pipe.predict_proba(X)[0], dtype=float)
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    classes = getattr(pipe, "classes_", None)
    if classes is None:
        try:
            classes = getattr(pipe.named_steps["model"], "classes_", None)
        except Exception:
            classes = None
    if classes is None:
        return None, "model_missing_classes_"
    classes = np.asarray(classes)
    try:
        p_phish = float(phish_probability_from_proba_row(proba, classes))
    except ValueError as e:
        return None, str(e)
    return p_phish, None


def _conservative_model_phishing_flag(by_name: Dict[str, Any]) -> bool:
    lr = by_name.get("logistic_regression")
    return bool(lr.get("predicted_phishing")) if lr is not None else False


def _primary_model_phishing_flag(by_name: Dict[str, Any], primary_ml: Dict[str, Any]) -> bool:
    rf = by_name.get("random_forest")
    if rf is not None:
        return bool(rf.get("predicted_phishing"))
    pp = Path(str(primary_ml.get("model_path") or "")).name.lower()
    primary_is_rf_family = "layer1_primary" in pp or "random_forest" in pp
    return bool(primary_ml.get("predicted_phishing")) if primary_is_rf_family else False


def _boosted_models_phishing_flag(by_name: Dict[str, Any]) -> bool:
    for k in ("xgboost", "lightgbm"):
        w = by_name.get(k)
        if w is not None and bool(w.get("predicted_phishing")):
            return True
    return False


def build_model_agreement_from_outputs(
    model_outputs: List[Dict[str, Any]],
    *,
    ml_primary_prob: Optional[float],
    primary_ml: Dict[str, Any],
) -> Dict[str, Any]:
    """Pure consensus + flags from optional witness rows (for tests and runtime)."""
    by_name = {str(m["model_name"]): m for m in model_outputs if m.get("model_name") is not None}
    n = len(model_outputs)
    votes_phish = sum(1 for m in model_outputs if m.get("predicted_phishing"))
    votes_legit = n - votes_phish
    q = _quorum_strong_votes(n)

    probs = [float(m["phish_probability"]) for m in model_outputs if isinstance(m.get("phish_probability"), (int, float))]
    spread = float(max(probs) - min(probs)) if len(probs) >= 2 else 0.0

    consensus: str
    if n < 2:
        return {
            "ml_primary_prob": ml_primary_prob,
            "ml_consensus": "unavailable",
            "ml_prob_spread": round(spread, 6) if probs else 0.0,
            "ml_model_votes_phishing": int(votes_phish),
            "ml_model_votes_legitimate": int(votes_legit),
            "ml_models_available": [str(m["model_name"]) for m in model_outputs],
            "ml_model_outputs": list(model_outputs),
            "conservative_model_phishing": _conservative_model_phishing_flag(by_name),
            "primary_model_phishing": _primary_model_phishing_flag(by_name, primary_ml),
            "boosted_models_phishing": _boosted_models_phishing_flag(by_name),
        }

    boosted_only = False
    lr = by_name.get("logistic_regression")
    rf = by_name.get("random_forest")
    if lr is not None and rf is not None:
        if not bool(lr.get("predicted_phishing")) and not bool(rf.get("predicted_phishing")):
            xgb = by_name.get("xgboost")
            lgb = by_name.get("lightgbm")
            if (xgb and bool(xgb.get("predicted_phishing"))) or (lgb and bool(lgb.get("predicted_phishing"))):
                boosted_only = True

    if boosted_only:
        consensus = "boosted_only"
    elif votes_phish >= q:
        consensus = "strong_phishing"
    elif votes_legit >= q:
        consensus = "strong_legitimate"
    elif votes_phish >= 1 and votes_legit >= 1:
        consensus = "split"
    else:
        consensus = "unavailable"

    return {
        "ml_primary_prob": ml_primary_prob,
        "ml_consensus": consensus,
        "ml_prob_spread": round(spread, 6) if probs else 0.0,
        "ml_model_votes_phishing": int(votes_phish),
        "ml_model_votes_legitimate": int(votes_legit),
        "ml_models_available": [str(m["model_name"]) for m in model_outputs],
        "ml_model_outputs": list(model_outputs),
        "conservative_model_phishing": _conservative_model_phishing_flag(by_name),
        "primary_model_phishing": _primary_model_phishing_flag(by_name, primary_ml),
        "boosted_models_phishing": _boosted_models_phishing_flag(by_name),
    }


def compute_layer1_model_agreement(
    url: str,
    primary_ml: Dict[str, Any],
    *,
    use_dns: bool = False,
) -> Dict[str, Any]:
    """
    Score optional witness pipelines in ``outputs/models``; never mutates primary scores.

    Primary probability stays on ``primary_ml``; this returns supporting agreement only.
    """
    try:
        p_cal = primary_ml.get("phish_proba_calibrated")
        p_raw = primary_ml.get("phish_proba")
        ml_primary: Optional[float] = None
        if isinstance(p_cal, (int, float)):
            ml_primary = float(p_cal)
        elif isinstance(p_raw, (int, float)):
            ml_primary = float(p_raw)
    except (TypeError, ValueError):
        ml_primary = None

    primary_path_str = str(primary_ml.get("model_path") or "")
    primary_resolved: Optional[Path] = None
    try:
        pprim = Path(primary_path_str)
        if pprim.is_file():
            primary_resolved = pprim.resolve()
    except OSError:
        primary_resolved = None

    model_outputs: List[Dict[str, Any]] = []
    try:
        X_raw, _, _ = build_layer1_frame(url, use_dns=use_dns)
    except Exception as e:  # noqa: BLE001
        logger.debug("model agreement feature build failed: %s", e)
        return {
            **build_model_agreement_from_outputs([], ml_primary_prob=ml_primary, primary_ml=primary_ml),
            "agreement_error": str(e),
        }

    expected = _expected_feature_columns()
    if expected:
        for c in expected:
            if c not in X_raw.columns:
                X_raw[c] = np.nan
        X = X_raw[expected]
    else:
        X = X_raw

    for logical_name, fname in _AGREEMENT_MODEL_FILES:
        path = models_dir() / fname
        if not path.is_file():
            continue
        try:
            if primary_resolved is not None and path.resolve() == primary_resolved:
                continue
        except OSError:
            pass
        try:
            pipe = joblib.load(path)
        except Exception as e:  # noqa: BLE001
            logger.debug("agreement model load failed %s: %s", path, e)
            continue
        p_phish, err = _raw_phish_proba_from_pipeline(pipe, X)
        if p_phish is None:
            logger.debug("agreement model score failed %s: %s", path, err)
            continue
        model_outputs.append(
            {
                "model_name": logical_name,
                "phish_probability": round(float(p_phish), 6),
                "predicted_phishing": bool(float(p_phish) >= 0.5),
            }
        )

    return build_model_agreement_from_outputs(model_outputs, ml_primary_prob=ml_primary, primary_ml=primary_ml)


def _default_model_path() -> Path:
    p = models_dir() / "layer1_primary.joblib"
    if p.is_file():
        return p
    return models_dir() / "logistic_regression.joblib"


def _calibrator_path() -> Path:
    return models_dir() / "layer1_probability_calibrator.joblib"


def _load_probability_calibrator() -> Optional[Dict[str, Any]]:
    p = _calibrator_path()
    if not p.is_file():
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        logger.debug("calibrator load failed: %s", e)
        return None


def _apply_probability_calibrator(p_raw: float, cal: Dict[str, Any]) -> float:
    kind = cal.get("type")
    model = cal.get("model")
    if model is None:
        return float(p_raw)
    x = float(p_raw)
    if kind == "isotonic":
        return float(np.clip(model.predict(np.array([x], dtype=float))[0], 0.0, 1.0))
    if kind == "platt":
        return float(np.clip(model.predict_proba(np.array([[x]], dtype=float))[0, 1], 0.0, 1.0))
    return float(x)


def _expected_feature_columns() -> Optional[List[str]]:
    cfg = reports_dir() / "training_config.json"
    if not cfg.is_file():
        return None
    try:
        meta = json.loads(cfg.read_text(encoding="utf-8"))
        cols = meta.get("feature_columns_used")
        if isinstance(cols, list) and cols:
            return cols
    except Exception as e:
        logger.debug("training_config read failed: %s", e)
    return None


def build_layer1_frame(
    url: str, *, use_dns: bool = False
) -> Tuple[pd.DataFrame, str, List[str]]:
    canon, inv, _ = canonicalize_url(url)
    if inv or not canon:
        canon = (url or "").strip()
    feats = extract_layer1_features(canon, use_dns=use_dns)
    brand_lines = explain_brand_structure_features(feats)
    feats.pop("canonical_url", None)
    return pd.DataFrame([feats]), canon, brand_lines


def predict_layer1(
    url: str,
    *,
    model_path: Optional[Path] = None,
    use_dns: bool = False,
) -> Dict[str, Any]:
    path = model_path or _default_model_path()
    if not path.is_file():
        return {
            "error": f"no_model_file:{path}",
            "phish_proba": None,
            "predicted_phishing": None,
            "brand_structure_explanations": [],
            "brand_structure_features": {},
        }
    pipe = joblib.load(path)
    X_raw, canon, brand_explain = build_layer1_frame(url, use_dns=use_dns)
    brand_snapshot = {
        k: int(X_raw.iloc[0][k]) if k in X_raw.columns and pd.notna(X_raw.iloc[0][k]) else 0
        for k in BRAND_STRUCTURE_FEATURE_KEYS
        if k in X_raw.columns
    }
    expected = _expected_feature_columns()
    if expected:
        for c in expected:
            if c not in X_raw.columns:
                X_raw[c] = np.nan
        X = X_raw[expected]
    else:
        X = X_raw
    try:
        proba = np.asarray(pipe.predict_proba(X)[0], dtype=float)
    except Exception as e:
        return {
            "error": str(e),
            "phish_proba": None,
            "predicted_phishing": None,
            "canonical_url": canon,
            "brand_structure_explanations": brand_explain,
            "brand_structure_features": brand_snapshot,
        }
    classes = getattr(pipe, "classes_", None)
    if classes is None:
        classes = getattr(pipe.named_steps["model"], "classes_", None)
    if classes is None:
        return {
            "error": "model_missing_classes_",
            "phish_proba": None,
            "predicted_phishing": None,
            "canonical_url": canon,
            "brand_structure_explanations": brand_explain,
            "brand_structure_features": brand_snapshot,
        }
    classes = np.asarray(classes)
    try:
        p_phish = phish_probability_from_proba_row(proba, classes)
    except ValueError as e:
        return {
            "error": str(e),
            "phish_proba": None,
            "predicted_phishing": None,
            "canonical_url": canon,
            "brand_structure_explanations": brand_explain,
            "brand_structure_features": brand_snapshot,
        }
    p_raw = float(p_phish)
    cal = _load_probability_calibrator()
    p_cal = _apply_probability_calibrator(p_raw, cal) if cal is not None else p_raw
    contrib: List[Dict[str, Any]] = []
    try:
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        X_t = prep.transform(X)
        if hasattr(model, "coef_"):
            names = prep.get_feature_names_out()
            coef = model.coef_.ravel()
            row = np.asarray(X_t).ravel()
            prod = np.abs(coef * row)
            top = np.argsort(-prod)[:12]
            for i in top:
                contrib.append(
                    {
                        "feature": str(names[i]),
                        "signed_coef": float(coef[i]),
                        "value": float(row[i]) if i < len(row) else None,
                    }
                )
    except Exception as e:
        logger.debug("contrib skipped: %s", e)
    return {
        "model_path": str(path),
        "canonical_url": canon,
        "phish_proba_model_raw": round(p_raw, 6),
        "phish_proba_calibrated": round(p_cal, 6),
        "phish_proba": round(p_cal, 6),
        "probability_calibration": (
            {"type": cal.get("type"), "path": str(_calibrator_path())} if cal is not None else None
        ),
        "predicted_phishing": bool(p_cal >= 0.5),
        "top_linear_signals": contrib,
        "brand_structure_explanations": brand_explain,
        "brand_structure_features": brand_snapshot,
        "model_classes_": [int(x) for x in classes.tolist()],
        "label_semantics": {"legitimate": 0, "phishing": 1, "source": "internal / Kaggle-mapped"},
    }
