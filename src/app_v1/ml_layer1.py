"""Load Layer-1 sklearn pipeline and score a single URL (fast URL/host features only)."""

from __future__ import annotations

import json
import logging
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
