"""Step 3: methodology checks, each retrained with the repo's own train.py on the SAME domain-grouped
split as step 1, written to separate variant folders (the original pipeline is not changed).

Variants
  asis            step 1 output (reference)
  scheme_neutral  every URL rewritten to https:// before feature extraction. In the Kaggle dump,
                  46% of phishing rows carry an explicit scheme vs 8% of legit rows, and bare hosts get
                  http:// during canonicalization, so https is a label proxy. train.py drops has_https
                  but keeps https_without_official_anchor / official_anchor_with_https and url_length.
  decontaminated  drops training rows whose registered domain appears in any evaluation list
                  (data/evaluation/url_suites.json, hard_legit_urls.jsonl, and the 8 audit URLs the
                  composite rule scores). simple_legit_augment adds those URLs to training, so the
                  suites and the composite audit were scored on training domains.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys

from common import REPO, WORK, isolate_env, seed_everything, write_result

isolate_env()
seed_everything()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phishguard import train as T  # noqa: E402
from phishguard.data.eval_sets import load_hard_legit_rows, load_url_suites  # noqa: E402
from phishguard.features.layer1 import extract_layer1_features  # noqa: E402
from phishguard.urls.safe import leak_safe_group_key  # noqa: E402

_spec = importlib.util.spec_from_file_location("ev", REPO / "metrics" / "scripts" / "02_evaluate_layer1.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

PROC = WORK / "data" / "processed"
VAR = WORK / "variants"
MODELS = ev.MODELS


def eval_domains() -> set:
    urls = [u for v in load_url_suites().values() for u in v]
    urls += [r["url"] for r in load_hard_legit_rows()]
    urls += list(T.LAYER1_OFFICIAL_HTTPS_AUDIT_URLS)
    return {leak_safe_group_key(u)[0] for u in urls}


def to_https(u: str) -> str:
    u = str(u or "")
    if "://" in u:
        return "https://" + u.split("://", 1)[1]
    return "https://" + u


def rebuild_features_https(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for u in df["canonical_url"].fillna("").astype(str):
        rows.append(extract_layer1_features(to_https(u), use_dns=False))
    feats = pd.DataFrame(rows).astype(str)
    keep = [c for c in df.columns if c not in feats.columns]
    out = pd.concat([df[keep].reset_index(drop=True), feats.drop(columns=["canonical_url"])], axis=1)
    out["canonical_url"] = feats["canonical_url"].values
    return out


def run_train(name: str, train_csv, test_csv) -> dict:
    vdir = VAR / name
    if vdir.exists():
        shutil.rmtree(vdir)
    (vdir / "outputs").mkdir(parents=True)
    env = dict(os.environ, PHISH_OUTPUTS_DIR=str(vdir / "outputs"), PHISH_LOGS_DIR=str(vdir / "logs"))
    subprocess.run(
        [sys.executable, "-m", "phishguard.models.train", "--train", str(train_csv), "--test", str(test_csv), "--layer1-only"],
        cwd=REPO, env=env, check=True,
    )
    return evaluate_variant(vdir / "outputs", test_csv, train_csv)


def evaluate_variant(outdir, test_csv, train_csv) -> dict:
    repo_metrics = json.loads((outdir / "metrics" / "metrics.json").read_text())
    tr = pd.read_csv(train_csv, dtype=str, low_memory=False)
    te = pd.read_csv(test_csv, dtype=str, low_memory=False)
    tr["_split"], te["_split"] = "train", "test"
    full = pd.concat([tr, te], ignore_index=True)
    full = full.loc[full["url_length"].notna()].reset_index(drop=True)
    X, y, _, _ = T._feature_matrix(full, T._exclude_layer1_only(full, True, include_dns=False))
    is_test = (full["_split"] == "test").values
    Xt, yt = X.loc[is_test].reset_index(drop=True), y[is_test]
    res = {"n_train_rows": int((~is_test).sum()), "n_test_rows": int(is_test.sum()), "n_features": int(X.shape[1]), "models": {}}
    for r in repo_metrics:
        m = r["model"]
        pipe = joblib.load(outdir / "models" / f"{m}.joblib")
        proba = ev.phish_proba(pipe, Xt)
        pm = ev.point_metrics(yt, pipe.predict(Xt), proba)
        pm["audit_official_https_mean_phish_proba"] = r.get("audit_official_https_mean_phish_proba")
        res["models"][m] = pm
    res["winner_composite"] = T._pick_layer1_primary_model_by_policy(repo_metrics, policy="composite")
    res["winner_by_f1"] = T._pick_layer1_primary_model_by_policy(repo_metrics, policy="f1")
    return res


def main() -> None:
    tr_csv, te_csv = PROC / "kaggle_train.csv", PROC / "kaggle_test.csv"
    out = {"command": "python metrics/scripts/03_ablations.py", "variants": {}}

    out["variants"]["asis"] = evaluate_variant(WORK / "outputs", te_csv, tr_csv)

    # Scheme-neutral: same rows, same split, features recomputed on https:// URLs.
    tr = pd.read_csv(tr_csv, dtype=str, low_memory=False)
    te = pd.read_csv(te_csv, dtype=str, low_memory=False)
    raw_scheme = tr["url"].fillna("").str.extract(r"^([a-zA-Z]+)://")[0].fillna("none").str.lower()
    out["scheme_artifact_in_train_rows"] = pd.crosstab(raw_scheme, tr["label"]).to_dict()
    sn_tr, sn_te = PROC / "scheme_neutral_train.csv", PROC / "scheme_neutral_test.csv"
    rebuild_features_https(tr).to_csv(sn_tr, index=False)
    rebuild_features_https(te).to_csv(sn_te, index=False)
    out["variants"]["scheme_neutral"] = run_train("scheme_neutral", sn_tr, sn_te)

    # Decontaminated: drop evaluation-list domains from TRAIN only; test set unchanged.
    dom = eval_domains()
    g = np.array([leak_safe_group_key(u)[0] for u in tr["canonical_url"].fillna("").astype(str)])
    mask = np.isin(g, list(dom))
    gt = np.array([leak_safe_group_key(u)[0] for u in te["canonical_url"].fillna("").astype(str)])
    out["decontamination"] = {
        "eval_registered_domains": sorted(dom),
        "train_rows_removed": int(mask.sum()),
        "train_rows_removed_by_source": tr.loc[mask, "source_dataset"].fillna("kaggle").value_counts().to_dict()
        if "source_dataset" in tr.columns else None,
        "test_rows_on_eval_domains": int(np.isin(gt, list(dom)).sum()),
    }
    dc_tr = PROC / "decontaminated_train.csv"
    tr.loc[~mask].to_csv(dc_tr, index=False)
    out["variants"]["decontaminated"] = run_train("decontaminated", dc_tr, te_csv)

    write_result("03_ablations", out)


if __name__ == "__main__":
    main()
