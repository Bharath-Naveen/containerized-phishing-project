"""Step 4: dashboard-level evaluation (ML-only path, reinforcement=False) of the curated URL suites,
the hard-legit false-positive list, the curated simple-legit list, and a random sample of held-out
Kaggle rows, with the legitimacy rescue layer ON and OFF. Also records Layer-1 latency.

Live Playwright capture (Layer 2) is NOT exercised here: the verification environment has no
general internet egress. Rescue-layer results therefore describe the ML-only path only.

Model variants (each scored in its own subprocess so model caches cannot leak between variants):
  deployed        outputs/models/layer1_primary.joblib (the model the app ships)
  asis_50k        step 1 composite winner (documented default run)
  decontaminated  step 3 composite winner trained without evaluation-list domains
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from common import REPO, SEED, WORK, isolate_env, seed_everything, write_result

isolate_env()
seed_everything()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VARIANT_OUTPUTS = {
    "deployed": REPO / "outputs",
    "asis_50k": WORK / "outputs",
    "decontaminated": WORK / "variants" / "decontaminated" / "outputs",
}
N_PER_CLASS = 200


def kaggle_sample(variant: str) -> list:
    """Held-out rows the variant never trained on: its own test split."""
    if variant == "deployed":
        te = pd.read_csv(REPO / "data" / "processed" / "retrain_with_fresh_test_20260427_234728.csv", dtype=str, low_memory=False)
    else:
        te = pd.read_csv(WORK / "data" / "processed" / "kaggle_test.csv", dtype=str, low_memory=False)
    te["label"] = pd.to_numeric(te["label"], errors="coerce").astype(int)
    parts = [te[te["label"] == k].sample(n=N_PER_CLASS, random_state=SEED) for k in (0, 1)]
    s = pd.concat(parts)
    return [{"url": u, "label": int(l), "set": "kaggle_heldout_sample"} for u, l in zip(s["canonical_url"], s["label"])]


def build_items(variant: str, include_simple: bool) -> list:
    from phishguard.data.eval_sets import load_hard_legit_rows, load_url_suites

    items = []
    for name, urls in load_url_suites().items():
        lab = 0 if "legit" in name else 1
        items += [{"url": u, "label": lab, "set": f"suite:{name}"} for u in urls]
    items += [{"url": r["url"], "label": 0, "set": "hard_legit"} for r in load_hard_legit_rows()]
    if include_simple:
        with open(REPO / "data" / "evaluation" / "simple_legit_urls.jsonl", encoding="utf-8") as f:
            items += [{"url": json.loads(l)["url"], "label": 0, "set": "simple_legit"} for l in f if l.strip()]
    rec = pd.read_csv(REPO / "phishing_dataset" / "dataset_test_recent.csv")
    items += [{"url": u, "label": 0, "set": "official_brand_2026_04_26"} for u in rec.loc[rec["source"] == "brand_official", "url"]]
    items += kaggle_sample(variant)
    return items


def worker(variant: str, include_simple: bool, out_path: str) -> None:
    from phishguard.app.dashboard import build_dashboard_analysis
    from phishguard.app.ml_layer1 import predict_layer1

    items = build_items(variant, include_simple)
    build_dashboard_analysis("https://example.com/", reinforcement=False)  # warm-up (model load)
    rows = []
    for it in items:
        t0 = time.perf_counter()
        ml = predict_layer1(it["url"])
        t_l1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        payload, _ = build_dashboard_analysis(it["url"], reinforcement=False)
        t_dash = time.perf_counter() - t0
        v = payload.get("verdict") or {}
        rows.append({
            **it,
            "p_raw": ml.get("phish_proba_model_raw"),
            "p_cal": ml.get("phish_proba_calibrated"),
            "l1_flag": bool(ml.get("predicted_phishing")),
            "verdict": v.get("verdict_3way"),
            "combined": v.get("combined_score"),
            "rescue_applied": bool(v.get("legitimacy_rescue_applied")),
            "t_layer1_s": t_l1,
            "t_dashboard_ml_only_s": t_dash,
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def summarize(rows: list) -> dict:
    df = pd.DataFrame(rows)
    mixed = df["set"] == "kaggle_heldout_sample"
    df.loc[mixed, "set"] = df.loc[mixed].apply(
        lambda r: "kaggle_heldout_sample:" + ("phishing" if r["label"] == 1 else "legit"), axis=1
    )
    out = {}
    for s, g in df.groupby("set"):
        lab = int(g["label"].iloc[0])
        vc = g["verdict"].value_counts().to_dict()
        d = {"n": int(len(g)), "expected": "phishing" if lab else "legitimate", "verdicts": vc}
        if lab == 1:
            d["pass_rate_likely_phishing"] = float(np.mean(g["verdict"] == "likely_phishing"))
            d["layer1_flag_rate"] = float(np.mean(g["l1_flag"]))
        else:
            d["pass_rate_strict_likely_legitimate"] = float(np.mean(g["verdict"] == "likely_legitimate"))
            d["pass_rate_not_flagged_phishing"] = float(np.mean(g["verdict"] != "likely_phishing"))
            d["false_positive_rate_likely_phishing"] = float(np.mean(g["verdict"] == "likely_phishing"))
            d["layer1_flag_rate"] = float(np.mean(g["l1_flag"]))
        d["rescue_applied_count"] = int(g["rescue_applied"].sum())
        out[s] = d
    return out


def latency(rows: list) -> dict:
    df = pd.DataFrame(rows)
    res = {}
    for col in ("t_layer1_s", "t_dashboard_ml_only_s"):
        ms = df[col].values * 1000
        res[col.replace("_s", "_ms")] = {"n": int(len(ms)), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)), "mean": float(ms.mean())}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--variant")
    ap.add_argument("--include-simple", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--summarize-only", action="store_true", help="Rebuild the result from existing worker files.")
    args = ap.parse_args()
    if args.worker:
        worker(args.variant, args.include_simple, args.out)
        return

    tmp = WORK / "step4"
    tmp.mkdir(parents=True, exist_ok=True)
    result = {"command": "python metrics/scripts/04_suites_and_fp.py", "mode": "ML-only dashboard path (reinforcement=False); no live capture", "variants": {}}
    for variant, outdir in VARIANT_OUTPUTS.items():
        result["variants"][variant] = {}
        for rescue in ("on", "off"):
            env = dict(os.environ, PHISH_OUTPUTS_DIR=str(outdir), PHISH_LEGITIMACY_RESCUE_ENABLED="true" if rescue == "on" else "false")
            outp = tmp / f"{variant}_rescue_{rescue}.json"
            cmd = [sys.executable, __file__, "--worker", "--variant", variant, "--out", str(outp)]
            if rescue == "on":
                cmd.append("--include-simple")
            if not args.summarize_only:
                print("running", variant, "rescue", rescue, flush=True)
                subprocess.run(cmd, cwd=REPO, env=env, check=True)
            rows = json.loads(outp.read_text())
            entry = {"per_set": summarize(rows)}
            if rescue == "on":
                entry["latency_ms_cpu"] = latency([r for r in rows if r["set"].startswith("kaggle_heldout_sample")])
            result["variants"][variant][f"rescue_{rescue}"] = entry
        on = pd.DataFrame(json.loads((tmp / f"{variant}_rescue_on.json").read_text()))
        off = pd.DataFrame(json.loads((tmp / f"{variant}_rescue_off.json").read_text()))
        on = on[on["set"] != "simple_legit"].reset_index(drop=True)
        changed = on[on["verdict"].values != off["verdict"].values]
        result["variants"][variant]["rescue_ablation"] = {
            "n_urls_compared": int(len(on)),
            "verdicts_changed": int(len(changed)),
            "legit_false_positives_rescue_on": int(((on["label"] == 0) & (on["verdict"] == "likely_phishing")).sum()),
            "legit_false_positives_rescue_off": int(((off["label"] == 0) & (off["verdict"] == "likely_phishing")).sum()),
        }
    write_result("04_suites_and_fp", result)


if __name__ == "__main__":
    main()
