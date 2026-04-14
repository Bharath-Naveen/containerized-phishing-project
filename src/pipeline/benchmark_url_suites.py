"""Benchmark Layer-1 (+ calibration) on curated URL suites (obvious/tricky legit, obvious/hard phish)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.app_v1.ml_layer1 import predict_layer1
from src.app_v1.verdict_policy import Verdict3WayConfig, verdict_3way
from src.pipeline.evaluation_sets import DEFAULT_URL_SUITES_JSON, load_url_suites


def _suite_label(name: str) -> int:
    n = name.lower()
    if "legit" in n:
        return 0
    if "phish" in n:
        return 1
    raise ValueError(f"Unknown suite bucket: {name}")


def score_suite(urls: List[str], y_true: int) -> Dict[str, Any]:
    raw_probs: List[float] = []
    cal_probs: List[float] = []
    verdicts: List[str] = []
    for u in urls:
        ml = predict_layer1(u.strip())
        pr = ml.get("phish_proba_model_raw")
        pc = ml.get("phish_proba_calibrated", pr)
        if pr is None:
            raw_probs.append(0.0)
            cal_probs.append(0.0)
            verdicts.append("uncertain")
            continue
        raw_probs.append(float(pr))
        cal_probs.append(float(pc) if pc is not None else float(pr))
        org = 0.0
        combined = min(1.0, max(0.0, 0.65 * float(cal_probs[-1]) + 0.35 * org))
        v, _ = verdict_3way(combined, Verdict3WayConfig())
        verdicts.append(v)
    y = np.full(len(cal_probs), y_true)
    p_cal = np.array(cal_probs, dtype=float)
    pred_bin = (p_cal >= 0.5).astype(int)
    out: Dict[str, Any] = {
        "n": len(urls),
        "mean_phish_proba_raw": float(np.mean(raw_probs)) if raw_probs else None,
        "mean_phish_proba_calibrated": float(np.mean(cal_probs)) if cal_probs else None,
        "fraction_likely_phishing_verdict_ml_only_proxy": sum(1 for v in verdicts if v == "likely_phishing")
        / max(len(verdicts), 1),
    }
    out["accuracy_threshold_0.5_cal"] = float(accuracy_score(y, pred_bin))
    out["precision_cal"] = float(precision_score(y, pred_bin, zero_division=0))
    out["recall_cal"] = float(recall_score(y, pred_bin, zero_division=0))
    out["f1_cal"] = float(f1_score(y, pred_bin, zero_division=0))
    try:
        out["roc_auc_cal"] = float(roc_auc_score(y, p_cal)) if len(np.unique(y)) > 1 else None
    except Exception:
        out["roc_auc_cal"] = None
    return out


def run_all_suites(path: Path | None = None) -> Dict[str, Any]:
    suites = load_url_suites(path)
    per: Dict[str, Any] = {}
    all_y: List[int] = []
    all_p: List[float] = []
    for name, urls in suites.items():
        yt = _suite_label(name)
        per[name] = score_suite(urls, yt)
        for u in urls:
            ml = predict_layer1(u.strip())
            pc = ml.get("phish_proba_calibrated", ml.get("phish_proba_model_raw"))
            if pc is None:
                continue
            all_y.append(yt)
            all_p.append(float(pc))
    overall: Dict[str, Any] = {}
    if all_y:
        y = np.array(all_y, dtype=int)
        p = np.array(all_p, dtype=float)
        pred = (p >= 0.5).astype(int)
        overall = {
            "n": len(y),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        }
    return {"per_suite": per, "overall_on_all_suites": overall}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Benchmark URL suites (requires trained layer1_primary).")
    ap.add_argument("--suites-json", type=Path, default=None)
    args = ap.parse_args()
    rep = run_all_suites(args.suites_json)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
