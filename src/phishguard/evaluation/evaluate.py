"""`phishguard evaluate`: every portfolio number from one command (rebuild Phase 3).

Reads a finished training run (the model bundle, the four fitted models and the saved
domain-grouped train/test split), measures everything a reviewer would ask about, and writes:

  metrics/results/evaluation.json   all numbers + provenance (commit, code trees, data hashes, env, seed)
  metrics/VERIFIED_METRICS.md       human-readable version
  docs/MODEL_CARD.md                model card generated from the same numbers

Nothing is typed by hand; the markdown is rendered from the JSON. Typical use:

    phishguard train --full          # train on all ~796K deduplicated Kaggle URLs (seed 42)
    phishguard evaluate --with-tests # measure it

Live-capture measurements (Playwright) are listed as NOT RUN here; they come from GitHub Actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from phishguard.config import SEED
from phishguard.paths import data_dir, outputs_dir, processed_dir, project_root, reports_dir

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]


# --------------------------------------------------------------------------- metrics helpers
def _counts(y: np.ndarray, pred: np.ndarray, w: Optional[np.ndarray] = None) -> Dict[str, float]:
    w = np.ones(len(y)) if w is None else w
    tp = float(np.sum(w * ((y == 1) & (pred == 1))))
    fp = float(np.sum(w * ((y == 0) & (pred == 1))))
    fn = float(np.sum(w * ((y == 1) & (pred == 0))))
    tn = float(np.sum(w * ((y == 0) & (pred == 0))))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1e-12), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def point_metrics(y: np.ndarray, proba: Optional[np.ndarray], threshold: float = 0.5,
                  pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if pred is None:
        pred = (proba >= threshold).astype(int)
    c = _counts(y, pred)
    out = {k: c[k] for k in ("precision", "recall", "f1", "false_positive_rate", "accuracy")}
    out["confusion_matrix"] = {"tn": int(c["tn"]), "fp": int(c["fp"]), "fn": int(c["fn"]), "tp": int(c["tp"])}
    if proba is not None and len(np.unique(proba)) > 1 and len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    else:
        out["roc_auc"], out["pr_auc"] = 0.5, float(np.mean(y))
    out["n"] = int(len(y))
    return out


def bootstrap_ci(y: np.ndarray, proba: np.ndarray, n_boot: int, threshold: float = 0.5) -> Dict[str, List[float]]:
    """Percentile 95% CI from n_boot row resamples (counts via Poisson-free multinomial weights)."""
    rng = np.random.default_rng(SEED)
    pred = (proba >= threshold).astype(int)
    keys = ["precision", "recall", "f1", "false_positive_rate", "roc_auc", "pr_auc"]
    acc: Dict[str, List[float]] = {k: [] for k in keys}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        w = np.bincount(idx, minlength=n).astype(float)
        c = _counts(y, pred, w)
        for k in ("precision", "recall", "f1", "false_positive_rate"):
            acc[k].append(c[k])
        yb, pb = y[idx], proba[idx]
        acc["roc_auc"].append(roc_auc_score(yb, pb))
        acc["pr_auc"].append(average_precision_score(yb, pb))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


def _phish_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    from phishguard.data.labels import phish_probability_from_proba_row

    P = pipe.predict_proba(X)
    cls = np.asarray(pipe.classes_)
    return np.array([phish_probability_from_proba_row(P[i], cls) for i in range(len(P))])


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- provenance
def provenance() -> Dict[str, Any]:
    from importlib import metadata

    root = project_root()
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, env=env).stdout.strip()
        except Exception:
            return "unknown"

    pkgs = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib", "tldextract", "pytest", "playwright"):
        try:
            pkgs[name] = metadata.version(name)
        except Exception:
            pkgs[name] = "not installed"
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_tracked_files": bool(git("-c", "core.fileMode=false", "status", "--porcelain", "--untracked-files=no")),
        "code_trees": {p: git("rev-parse", f"HEAD:{p}") for p in ("src", "tests", "data/evaluation")},
        "seed": SEED,
        "environment": {"python": sys.version.split()[0], "os": f"{platform.system()} {platform.release()}",
                        "cpu": cpu, "cpu_count": os.cpu_count(), "gpu": "none (CPU only)", "packages": pkgs},
    }


# --------------------------------------------------------------------------- sections
def run_tests() -> Dict[str, Any]:
    import xml.etree.ElementTree as ET
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="phishguard_tests_"))
    junit, cov = tmp / "junit.xml", tmp / "coverage.json"
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", f"--junitxml={junit}"]
    has_cov = subprocess.run([sys.executable, "-c", "import pytest_cov"], capture_output=True).returncode == 0
    if has_cov:
        cmd += ["--cov=src/phishguard", f"--cov-report=json:{cov}"]
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=project_root(), capture_output=True, text=True,
                   env=dict(os.environ, COVERAGE_FILE=str(tmp / ".coverage")))
    root = ET.parse(junit).getroot()
    s = root if root.tag == "testsuite" else root.find("testsuite")
    n, f, e, sk = (int(s.get(k)) for k in ("tests", "failures", "errors", "skipped"))
    out = {"collected": n, "passed": n - f - e - sk, "failed": f, "errors": e, "skipped": sk,
           "pass_rate": (n - f - e - sk) / n if n else None, "runtime_seconds": round(time.perf_counter() - t0, 1),
           "command": "pytest"}
    if has_cov and cov.is_file():
        tot = json.loads(cov.read_text())["totals"]
        out["line_coverage_percent"] = round(tot["percent_covered"], 1)
    return out


def data_section(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    rep = reports_dir()

    def load(name: str) -> Dict[str, Any]:
        p = rep / name
        return json.loads(p.read_text()) if p.is_file() else {}

    manifest = load("kaggle_sample_manifest.json")
    clean = load("kaggle_clean_stats.json")
    raw_csv = next(iter(sorted((data_dir() / "raw" / "kaggle").glob("*.csv"))), None)
    if raw_csv is None:
        raw_csv = project_root() / "data" / "raw" / "kaggle" / "phishing_and_legitimate_urls.csv"
    raw = {}
    if raw_csv.is_file():
        r = pd.read_csv(raw_csv, dtype=str, usecols=[0, 1])
        raw = {"rows": int(len(r)), "columns": int(pd.read_csv(raw_csv, nrows=1).shape[1]),
               "legit_status_1": int((r.iloc[:, 1] == "1").sum()), "phishing_status_0": int((r.iloc[:, 1] == "0").sum()),
               "sha256": _sha256(raw_csv), "file": "data/raw/kaggle/" + raw_csv.name,
               "source": "Kaggle harisudhan411/phishing-and-legitimate-urls", "date_range": "not available (no date column)"}
    return {
        "raw": raw,
        "deduplicated_rows": clean.get("canonical_url_rows_after"),
        "run_mode": manifest.get("run_mode"),
        "curated_legit_added": (manifest.get("simple_legit_augment") or {}).get("rows_added"),
        "tranco_rows_added": (manifest.get("tranco_augment") or {}).get("rows_added_after_dedupe"),
        "evaluation_exclusions": load("evaluation_exclusions.json"),
        "split": load("split_leak_safe_stats.json"),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_class_counts": {"legit": int((train["label"] == 0).sum()), "phishing": int((train["label"] == 1).sum())},
        "test_class_counts": {"legit": int((test["label"] == 0).sum()), "phishing": int((test["label"] == 1).sum())},
        "train_csv_sha256": _sha256(processed_dir() / "kaggle_train.csv"),
        "test_csv_sha256": _sha256(processed_dir() / "kaggle_test.csv"),
    }


def external_url_frame(urls: List[str], cols: List[str], ref: pd.DataFrame) -> pd.DataFrame:
    from phishguard.features.layer1 import extract_layer1_features

    X = pd.DataFrame([extract_layer1_features(u) for u in urls]).reindex(columns=cols)
    for c in cols:
        if pd.api.types.is_numeric_dtype(ref[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        else:
            X[c] = X[c].fillna("missing").astype(str)
    return X


def dashboard_verdicts(urls: List[str]) -> Dict[str, Any]:
    """Real dashboard (ML-only path) for each URL; also times Layer-1 and the full ML-only analysis."""
    from phishguard.app.dashboard import build_dashboard_analysis
    from phishguard.app.ml_layer1 import predict_layer1

    rows = []
    for u in urls:
        t0 = time.perf_counter()
        ml = predict_layer1(u)
        t1 = time.perf_counter()
        payload, _ = build_dashboard_analysis(u, reinforcement=False)
        t2 = time.perf_counter()
        rows.append({"url": u, "verdict": payload["verdict"]["verdict_3way"],
                     "layer1_flag": bool(ml.get("predicted_phishing")), "t_l1": t1 - t0, "t_dash": t2 - t1})
    return {"rows": rows}


def _verdict_summary(rows: List[Dict[str, Any]], expected: str) -> Dict[str, Any]:
    v = [r["verdict"] for r in rows]
    out = {"n": len(v), "verdicts": {k: v.count(k) for k in ("likely_phishing", "uncertain", "likely_legitimate")},
           "layer1_flag_rate": float(np.mean([r["layer1_flag"] for r in rows])) if rows else None}
    if expected == "phishing":
        out["detected_likely_phishing_rate"] = v.count("likely_phishing") / len(v) if v else None
    else:
        out["false_alarm_likely_phishing_rate"] = v.count("likely_phishing") / len(v) if v else None
        out["confirmed_likely_legitimate_rate"] = v.count("likely_legitimate") / len(v) if v else None
    return out


# --------------------------------------------------------------------------- main evaluation
def evaluate(*, n_boot: int = 1000, cv_rows: int = 100_000, n_dashboard: int = 400, with_tests: bool = False,
             results_dir: Optional[Path] = None) -> Dict[str, Any]:
    from phishguard.app.ml_layer1 import load_layer1_bundle
    from phishguard.data.eval_sets import (
        load_hard_legit_rows, load_official_brand_rows, load_phishstats_rows, load_url_suites,
    )
    from phishguard.models.train import _exclude_layer1_only, _feature_matrix
    from phishguard.urls.normalize import canonical_url
    from phishguard.urls.safe import leak_safe_group_key, safe_hostname

    t_start = time.perf_counter()
    bundle = load_layer1_bundle()
    if bundle is None:
        raise SystemExit(f"No model bundle in {outputs_dir() / 'models'}. Run `phishguard train` first.")
    report: Dict[str, Any] = {"provenance": provenance(), "not_run": []}

    tr = pd.read_csv(processed_dir() / "kaggle_train.csv", dtype=str, low_memory=False)
    te = pd.read_csv(processed_dir() / "kaggle_test.csv", dtype=str, low_memory=False)
    tr["_split"], te["_split"] = "train", "test"
    full = pd.concat([tr, te], ignore_index=True)
    full = full.loc[full["url_length"].notna() & (full["url_length"].astype(str).str.len() > 0)].reset_index(drop=True)
    X, y, num_cols, _ = _feature_matrix(full, _exclude_layer1_only(full, True, include_dns=False))
    cols = bundle["feature_columns"]
    assert list(X.columns) == cols, "feature columns differ from the bundle; evaluate the run that produced it"
    is_te = (full["_split"] == "test").values
    Xt, yt = X.loc[is_te].reset_index(drop=True), y[is_te]
    full["label"] = y
    report["data"] = data_section(full.loc[~is_te], full.loc[is_te])
    report["data"]["n_model_features"] = len(cols)
    tcfg_p = reports_dir() / "training_config.json"
    tcfg = json.loads(tcfg_p.read_text()) if tcfg_p.is_file() else {}
    report["data"]["feature_parity_check"] = tcfg.get("feature_parity_check")
    report["data"]["n_fit_rows"] = tcfg.get("n_fit_rows")
    report["data"]["n_validation_rows"] = tcfg.get("n_validation_rows")
    print("data loaded", flush=True)

    # held-out test, every model + majority baseline
    mdir = outputs_dir() / "models"
    pipes = {m: joblib.load(mdir / f"{m}.joblib") for m in MODEL_NAMES if (mdir / f"{m}.joblib").is_file()}
    run_metrics = json.loads((outputs_dir() / "metrics" / "metrics.json").read_text())
    val = {m["model"]: {k: m.get(k) for k in ("val_pr_auc", "val_roc_auc", "val_f1", "val_official_brand_fp_rate")} for m in run_metrics}
    majority = int(round(np.mean(y[~is_te])))
    held = {"majority_class_baseline": point_metrics(yt, None, pred=np.full(len(yt), majority))}
    probas = {}
    for m, pipe in pipes.items():
        p = _phish_proba(pipe, Xt)
        probas[m] = p
        held[m] = {**point_metrics(yt, p), "validation": val.get(m, {})}
        if n_boot:
            held[m]["bootstrap_95ci"] = bootstrap_ci(yt, p, n_boot)
        print("scored", m, flush=True)
    primary = bundle["model_name"]
    cal = bundle.get("calibrator")
    p_raw = _phish_proba(bundle["pipeline"], Xt)
    p_cal = np.clip(cal["model"].predict(p_raw), 0, 1) if cal and cal.get("type") == "isotonic" else p_raw
    report["held_out_test"] = {
        "threshold": "raw P(phishing) >= 0.5 (sklearn default); primary also shown at the app threshold",
        "test_rows": int(len(yt)), "models": held,
    }
    report["primary"] = {
        "model": primary, "selection_policy": bundle.get("selection_policy"),
        "selection_rule": "highest validation PR-AUC (domain-grouped validation split of the training rows); tie-break: fewer false alarms on the official-brand validation half",
        "bundle_created_utc": bundle.get("created_utc"), "bundle_train_csv_sha256": bundle.get("train_csv_sha256"),
        "app_threshold_calibrated_ge_0_5": point_metrics(yt, p_cal),
        "calibration": {"type": cal.get("type") if cal else None,
                        "brier_raw": float(np.mean((p_raw - yt) ** 2)), "brier_calibrated": float(np.mean((p_cal - yt) ** 2))},
    }

    # grouped 5-fold CV on a stratified subsample
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y)) if cv_rows <= 0 or cv_rows >= len(y) else np.sort(rng.choice(len(y), cv_rows, replace=False))
    Xc, yc = X.iloc[idx].reset_index(drop=True), y[idx]
    gc = np.array([leak_safe_group_key(u)[0] for u in full["canonical_url"].iloc[idx].fillna("").astype(str)])
    folds = {m: [] for m in ["majority_class_baseline", *pipes]}
    overlap = []
    for k, (a, b) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xc, yc, gc)):
        overlap.append(len(set(gc[a]) & set(gc[b])))
        maj = int(round(np.mean(yc[a])))
        folds["majority_class_baseline"].append(point_metrics(yc[b], None, pred=np.full(len(b), maj)))
        for m, pipe in pipes.items():
            f = clone(pipe).fit(Xc.iloc[a], yc[a])
            folds[m].append(point_metrics(yc[b], _phish_proba(f, Xc.iloc[b])))
        print(f"cv fold {k + 1}/5", flush=True)
    report["cross_validation"] = {
        "method": "StratifiedGroupKFold(5) by registered domain", "rows": int(len(idx)),
        "group_overlap_per_fold": overlap,
        "models": {m: {key: {"mean": float(np.mean([f[key] for f in fs])), "std": float(np.std([f[key] for f in fs], ddof=1))}
                       for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate")}
                   for m, fs in folds.items()},
    }

    # external evaluation sets (never in training)
    train_domains = set(full.loc[~is_te, "canonical_url"].fillna("").map(lambda u: leak_safe_group_key(u)[0]))
    train_hosts = set(full.loc[~is_te, "canonical_url"].fillna("").map(lambda u: safe_hostname(canonical_url(u)[0])[0]))
    ext: Dict[str, Any] = {}
    for name, rows, label in (
        ("official_brand_test_half", load_official_brand_rows(split="test"), 0),
        ("official_brand_validation_half", load_official_brand_rows(split="val"), 0),
        ("phishstats", load_phishstats_rows(), 1),
    ):
        urls = [r["url"] for r in rows]
        Xe = external_url_frame(urls, cols, X)
        per_model = {}
        for m, pipe in pipes.items():
            pe = _phish_proba(pipe, Xe)
            per_model[m] = float(np.mean(pe >= 0.5))
        pe_raw = _phish_proba(bundle["pipeline"], Xe)
        pe_cal = np.clip(cal["model"].predict(pe_raw), 0, 1) if cal and cal.get("type") == "isotonic" else pe_raw
        ext[name] = {
            "n": len(urls), "label": "phishing" if label else "legitimate",
            "layer1_flag_rate_by_model_raw_0_5": per_model,
            "primary_flag_rate_app_threshold": float(np.mean(pe_cal >= 0.5)),
            "urls_whose_domain_is_in_training": int(sum(leak_safe_group_key(u)[0] in train_domains for u in urls)),
            "urls_whose_host_is_in_training": int(sum(safe_hostname(canonical_url(u)[0])[0] in train_hosts for u in urls)),
        }
        if name == "official_brand_validation_half":
            ext[name]["note"] = "used to break ties when selecting the primary model; report the test half"
    report["external_sets"] = ext
    print("external sets done", flush=True)

    # dashboard verdicts (ML-only path) + latency
    dash: Dict[str, Any] = {"mode": "ML-only (reinforcement=False): no live page capture", "sets": {}}
    sets = {
        "official_brand_test_half": ([r["url"] for r in load_official_brand_rows(split="test")], "legitimate"),
        "phishstats": ([r["url"] for r in load_phishstats_rows()], "phishing"),
        "hard_legit": ([r["url"] for r in load_hard_legit_rows()], "legitimate"),
    }
    for sname, urls in load_url_suites().items():
        sets[f"suite:{sname}"] = (urls, "legitimate" if "legit" in sname else "phishing")
    te_rows = full.loc[is_te, ["canonical_url", "label"]].reset_index(drop=True)
    half = n_dashboard // 2
    for lab, nm in ((0, "heldout_test_sample_legit"), (1, "heldout_test_sample_phishing")):
        pool = te_rows[te_rows["label"] == lab]
        sets[nm] = (pool.sample(n=min(half, len(pool)), random_state=SEED)["canonical_url"].tolist(),
                    "phishing" if lab else "legitimate")
    timing_l1, timing_dash = [], []
    for sname, (urls, expected) in sets.items():
        res = dashboard_verdicts(urls)["rows"]
        dash["sets"][sname] = _verdict_summary(res, expected)
        if sname.startswith("heldout_test_sample"):
            timing_l1 += [r["t_l1"] for r in res]
            timing_dash += [r["t_dash"] for r in res]
        print("dashboard", sname, flush=True)
    report["dashboard_ml_only"] = dash
    ms1, ms2 = np.array(timing_l1) * 1000, np.array(timing_dash) * 1000
    report["latency_ms"] = {
        "n_urls": int(len(ms1)), "hardware": report["provenance"]["environment"]["cpu"],
        "layer1_only": {"p50": float(np.percentile(ms1, 50)), "p95": float(np.percentile(ms1, 95))},
        "dashboard_ml_only_incl_eal": {"p50": float(np.percentile(ms2, 50)), "p95": float(np.percentile(ms2, 95))},
        "note": "warm process, one URL at a time, includes 4-model agreement",
    }

    report["not_run"] = [
        {"item": "Full-path latency with Playwright live capture", "reason": "needs internet access to arbitrary sites; runs in GitHub Actions (Phase 4)"},
        {"item": "Legitimacy-rescue false-positive delta with live capture", "reason": "rescue conditions need capture evidence; Phase 4"},
        {"item": "Live EAL edge-case suite", "reason": "needs live capture; Phase 4"},
    ]
    if with_tests:
        report["tests"] = run_tests()
    report["runtime_seconds"] = round(time.perf_counter() - t_start, 1)

    out_dir = results_dir or (project_root() / "metrics" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from phishguard.evaluation.report import write_markdown

    write_markdown(report, project_root() / "metrics" / "VERIFIED_METRICS.md", project_root() / "docs" / "MODEL_CARD.md")
    print("wrote", out_dir / "evaluation.json")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure a trained run and write verified metrics + model card.")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for 95%% CIs (0 to skip)")
    ap.add_argument("--cv-rows", type=int, default=100_000, help="rows for grouped 5-fold CV (0 = all rows)")
    ap.add_argument("--n-dashboard", type=int, default=400, help="held-out URLs run through the dashboard (half legit, half phishing)")
    ap.add_argument("--with-tests", action="store_true", help="also run pytest and record counts + coverage")
    ap.add_argument("--results-dir", type=Path, default=None)
    a = ap.parse_args()
    evaluate(n_boot=a.n_boot, cv_rows=a.cv_rows, n_dashboard=a.n_dashboard, with_tests=a.with_tests, results_dir=a.results_dir)


if __name__ == "__main__":
    main()
