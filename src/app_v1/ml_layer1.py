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
from src.pipeline.layer1_features import extract_layer1_features
from src.pipeline.paths import models_dir, reports_dir

logger = logging.getLogger(__name__)


def _default_model_path() -> Path:
    p = models_dir() / "layer1_primary.joblib"
    if p.is_file():
        return p
    return models_dir() / "logistic_regression.joblib"


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


def build_layer1_frame(url: str, *, use_dns: bool = False) -> Tuple[pd.DataFrame, str]:
    canon, inv, _ = canonicalize_url(url)
    if inv or not canon:
        canon = (url or "").strip()
    feats = extract_layer1_features(canon, use_dns=use_dns)
    feats.pop("canonical_url", None)
    return pd.DataFrame([feats]), canon


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
        }
    pipe = joblib.load(path)
    X_raw, canon = build_layer1_frame(url, use_dns=use_dns)
    expected = _expected_feature_columns()
    if expected:
        for c in expected:
            if c not in X_raw.columns:
                X_raw[c] = np.nan
        X = X_raw[expected]
    else:
        X = X_raw
    try:
        proba = pipe.predict_proba(X)[0]
    except Exception as e:
        return {"error": str(e), "phish_proba": None, "predicted_phishing": None, "canonical_url": canon}
    classes = list(getattr(pipe.named_steps["model"], "classes_", [0, 1]))
    try:
        idx1 = classes.index(1)
        p_phish = float(proba[idx1])
    except ValueError:
        p_phish = float(proba[-1])
    pred = int(pipe.predict(X)[0])
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
        "phish_proba": round(p_phish, 6),
        "predicted_phishing": bool(pred == 1),
        "top_linear_signals": contrib,
    }
