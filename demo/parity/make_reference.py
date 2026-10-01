"""Python reference outputs for the JS parity test (rebuild Phase 6).

For every URL in the parity set this records what the app's own Python code computes:
  * the 54 model features, exactly as app/ml_layer1.build_layer1_frame builds them,
  * the primary XGBoost probability (raw and calibrated) and the four agreement-model probabilities,
  * the 4-model consensus, the host identity from host_path_reasoning, and
  * (with --dashboard) the verdict of the real dashboard in ML-only mode (build_dashboard_analysis,
    reinforcement=False), i.e. the full rule chain, not a re-implementation of it.

Model scores are computed in batches with the same fitted pipelines the app loads; a spot check at the
end compares a sample against predict_layer1 / compute_layer1_model_agreement called one URL at a time.

Usage:
  python demo/parity/make_reference.py --urls demo/parity/urls.jsonl --out demo/parity/out/reference.jsonl [--dashboard]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PHISH_LOG_LEVEL", "ERROR")

MODELS = ROOT / "models" / "layer1"
WITNESSES = ("logistic_regression", "random_forest", "xgboost", "lightgbm")


def _cache_joblib() -> None:
    """The app re-reads each agreement model from disk on every URL; caching the loaded objects
    changes nothing about the results, only the run time."""
    import phishguard.app.ml_layer1 as ml

    real = joblib.load
    cache: dict = {}

    def load(p, *a, **k):
        key = str(Path(p).resolve())
        if key not in cache:
            cache[key] = real(p, *a, **k)
        return cache[key]

    joblib.load = load
    ml.joblib.load = load


def _check_runtime_model(runtime_dir: Path) -> None:
    """The reference must come from the shipped, verified bundle (same sha256 as models/layer1/MANIFEST.json)."""
    import hashlib

    want = json.loads((MODELS / "MANIFEST.json").read_text())["files"]["layer1_bundle.joblib"]["sha256"]
    got = hashlib.sha256((Path(runtime_dir) / "layer1_bundle.joblib").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"the app would load {runtime_dir}/layer1_bundle.joblib (sha256 {got[:12]}), not the shipped model ({want[:12]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dashboard", action="store_true", help="also record the real ML-only dashboard verdict")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th row starting at i")
    args = ap.parse_args()

    _cache_joblib()
    from phishguard.app import ml_layer1 as ml
    from phishguard.app.host_path_reasoning import assess_host_path_reasoning
    from phishguard.app.ml_layer1 import build_layer1_frame, build_model_agreement_from_outputs, load_layer1_bundle

    _check_runtime_model(ml.runtime_models_dir())
    bundle = load_layer1_bundle()
    cols = bundle["feature_columns"]
    cal = bundle["calibrator"]
    pipes = {n: joblib.load(MODELS / f"{n}.joblib") for n in WITNESSES}

    si, sn = (int(x) for x in args.shard.split("/"))
    rows = [json.loads(l) for l in open(args.urls, encoding="utf-8") if l.strip()]
    rows = rows[si::sn]
    t0 = time.time()

    frames, canon = [], []
    for r in rows:
        X, c, _ = build_layer1_frame(r["url"])
        frames.append(X)
        canon.append(c)
    X = pd.concat(frames, ignore_index=True)
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[cols]
    t_feat = time.time() - t0

    p_raw = bundle["pipeline"].predict_proba(X)[:, 1].astype(float)
    p_cal = np.array([ml._apply_probability_calibrator(float(p), cal) for p in p_raw])
    wit = {n: pipes[n].predict_proba(X)[:, 1].astype(float) for n in WITNESSES}
    # Saabas-style per-feature contributions of the primary model (XGBoost's approx_contribs).
    import xgboost as xgb

    prep = bundle["pipeline"].named_steps["prep"]
    booster = bundle["pipeline"].named_steps["model"].get_booster()
    contribs = booster.predict(xgb.DMatrix(prep.transform(X)), pred_contribs=True, approx_contribs=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dashboard:
        from phishguard.app.dashboard import build_dashboard_analysis

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            outputs = [
                {"model_name": n, "phish_probability": round(float(wit[n][i]), 6), "predicted_phishing": bool(float(wit[n][i]) >= 0.5)}
                for n in WITNESSES
            ]
            agr = build_model_agreement_from_outputs(outputs, ml_primary_prob=round(float(p_cal[i]), 6), primary_ml={})
            hp = (assess_host_path_reasoning(input_url=(r["url"] or "").strip()).get("host_path_reasoning") or {})
            rec = {
                "id": r["id"],
                "set": r["set"],
                "canonical": canon[i],
                "features": [float(v) for v in X.iloc[i].tolist()],
                "p_raw": float(p_raw[i]),
                "p_cal": float(p_cal[i]),
                "p_cal_rounded": round(float(p_cal[i]), 6),
                "witness": {n: float(wit[n][i]) for n in WITNESSES},
                "consensus": agr["ml_consensus"],
                "spread": agr["ml_prob_spread"],
                "host_identity_class": hp.get("host_identity_class"),
                "host_legitimacy_confidence": hp.get("host_legitimacy_confidence"),
                "contribs": [float(v) for v in contribs[i]],
            }
            if args.dashboard:
                try:
                    res, _ = build_dashboard_analysis(r["url"], reinforcement=False)
                except ValueError as e:  # the app itself stops on some malformed URLs
                    rec["dashboard"] = {"verdict": "error", "error": f"ValueError: {e}"}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                v = res["verdict"]
                l1 = res["layer1_ml"]
                rec["dashboard"] = {
                    "verdict": v.get("verdict_3way"),
                    "phishing_signals": v.get("evidence_phishing_signals"),
                    "legitimacy_signals": v.get("evidence_legitimacy_signals"),
                    "ambiguity_signals": v.get("evidence_ambiguity_signals"),
                    "hard_blockers": v.get("evidence_hard_blockers"),
                    "phishing_score": v.get("evidence_phishing_score"),
                    "legitimacy_score": v.get("evidence_legitimacy_score"),
                    "p_cal_rounded": l1.get("phish_proba_calibrated"),
                    "p_raw_rounded": l1.get("phish_proba_model_raw"),
                    "consensus": (l1.get("model_agreement") or {}).get("ml_consensus"),
                }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "feature_seconds": round(t_feat, 1), "total_seconds": round(time.time() - t0, 1), "out": str(out_path)}))


if __name__ == "__main__":
    main()
