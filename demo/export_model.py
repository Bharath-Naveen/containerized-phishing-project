"""Export the shipped Layer-1 models to compact files the browser demo can evaluate.

Rebuild Phase 6. Reads only the verified model files in ``models/layer1/`` (hash-checked against
MANIFEST.json) and the public suffix list tldextract is using, and writes:

  demo/build/model-core.json    scalers, XGBoost (primary) trees, its isotonic calibrator,
                                LightGBM trees, Logistic Regression coefficients, the ICANN
                                public suffix rules, and provenance (hashes of every input)
  demo/build/model-rf.bin.gz    the Random Forest, packed (loaded only when the demo needs it)

Nothing is re-fit or approximated: thresholds and leaf values are exported so that the JS
evaluator makes exactly the same comparisons the Python libraries make (see notes per model).

Usage:  python demo/export_model.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = ROOT / "models" / "layer1"
OUT = ROOT / "demo" / "build"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_manifest() -> dict:
    man = json.loads((MODELS / "MANIFEST.json").read_text())
    for name, meta in man["files"].items():
        got = sha256(MODELS / name)
        if got != meta["sha256"]:
            raise SystemExit(f"{name}: sha256 {got} does not match MANIFEST.json")
    return man


def scaler_of(pipe) -> list:
    ct = pipe.named_steps["prep"]
    assert len(ct.transformers_) == 1 and ct.remainder == "drop"
    num = ct.transformers_[0][1]
    imp, sc = num.named_steps["imputer"], num.named_steps["scaler"]
    assert sc.with_mean is False
    # The imputer only matters for NaN inputs; Layer-1 features are never NaN.
    return [float(x) for x in sc.scale_], [float(x) for x in imp.statistics_]


def floor_f32(t: float) -> float:
    """Largest float32 <= t. For a float32 input x: (x <= t) == (x <= floor_f32(t))."""
    f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, np.float32(-np.inf))
    return float(f)


def export_xgboost(pipe) -> dict:
    model = pipe.named_steps["model"]
    raw = json.loads(model.get_booster().save_raw("json"))
    learner = raw["learner"]
    assert learner["objective"]["name"] == "binary:logistic"
    base_score = float(learner["learner_model_param"]["base_score"])
    # XGBoost stores base_score as a probability and turns it into a float32 margin.
    # Same float32 arithmetic as XGBoost's ProbToMargin: -log(1/p - 1).
    p = np.float32(base_score)
    base_margin = float(-np.log(np.float32(1.0) / p - np.float32(1.0)))
    trees = []
    for t in learner["gradient_booster"]["model"]["trees"]:
        assert all(c == 0 for c in t["split_type"]), "categorical splits not supported"
        trees.append(
            {
                "l": t["left_children"],
                "r": t["right_children"],
                "f": t["split_indices"],
                # For leaves split_conditions holds the leaf value. All are float32 values.
                "v": [float(np.float32(x)) for x in t["split_conditions"]],
                "d": t["default_left"],
                "c": [float(np.float32(x)) for x in t["sum_hessian"]],
            }
        )
    tree_info = learner["gradient_booster"]["model"]["tree_info"]
    assert all(x == 0 for x in tree_info)
    return {"base_score": base_score, "base_margin_f32": base_margin, "trees": trees}


def export_lightgbm(pipe) -> dict:
    model = pipe.named_steps["model"]
    d = model.booster_.dump_model()
    assert d["objective"].startswith("binary sigmoid:")
    sig = float(d["objective"].split(":")[1])
    trees = []
    for ti in d["tree_info"]:
        nodes = {"f": [], "t": [], "dl": [], "mt": [], "l": [], "r": [], "v": []}

        def walk(n):
            idx = len(nodes["f"])
            for k in nodes:
                nodes[k].append(None)
            if "leaf_value" in n or "split_feature" not in n:
                nodes["f"][idx] = -1
                nodes["v"][idx] = float(n["leaf_value"])
                return idx
            assert n["decision_type"] == "<="
            nodes["f"][idx] = int(n["split_feature"])
            nodes["t"][idx] = float(n["threshold"])
            nodes["dl"][idx] = bool(n["default_left"])
            nodes["mt"][idx] = {"None": 0, "Zero": 1, "NaN": 2}[n["missing_type"]]
            nodes["l"][idx] = walk(n["left_child"])
            nodes["r"][idx] = walk(n["right_child"])
            return idx

        walk(ti["tree_structure"])
        trees.append(nodes)
    return {"sigmoid": sig, "trees": trees}


def export_logreg(pipe) -> dict:
    m = pipe.named_steps["model"]
    assert m.coef_.shape[0] == 1
    return {"coef": [float(x) for x in m.coef_[0]], "intercept": float(m.intercept_[0])}


def export_random_forest(pipe) -> tuple[bytes, dict]:
    """Columnar packing: per internal node (feature u8, threshold index u16, right-child offset u16);
    per leaf the phishing fraction as float64 (exact). Left child is always the next node (sklearn
    builds depth-first), which the exporter asserts. Thresholds are stored per feature as float32
    values rounded down, which keeps every comparison identical for float32 inputs."""
    rf = pipe.named_steps["model"]
    assert list(rf.classes_) == [0, 1]
    n_feat = rf.n_features_in_
    thr_tables: list[dict] = [dict() for _ in range(n_feat)]
    kinds, feats, thr_idx, rights, leaves, tree_sizes = [], [], [], [], [], []
    for est in rf.estimators_:
        t = est.tree_
        n = t.node_count
        tree_sizes.append(n)
        assert n < 65536
        for i in range(n):
            left, right = t.children_left[i], t.children_right[i]
            if left == -1:
                kinds.append(1)
                v = t.value[i][0]
                frac = float(v[1])
                # value is already normalised by sklearn; keep the exact float64.
                leaves.append(frac)
            else:
                assert left == i + 1, "expected depth-first node order"
                kinds.append(0)
                f = int(t.feature[i])
                th = floor_f32(float(t.threshold[i]))
                tab = thr_tables[f]
                if th not in tab:
                    tab[th] = len(tab)
                feats.append(f)
                thr_idx.append(tab[th])
                rights.append(int(right) - i)
    assert all(len(t) < 65536 for t in thr_tables)
    tables = [sorted(t, key=t.get) for t in thr_tables]
    parts = [
        np.asarray(tree_sizes, dtype="<u2").tobytes(),
        np.asarray(kinds, dtype="u1").tobytes(),
        np.asarray(feats, dtype="u1").tobytes(),
        np.asarray(thr_idx, dtype="<u2").tobytes(),
        np.asarray(rights, dtype="<u2").tobytes(),
        np.asarray(leaves, dtype="<f8").tobytes(),
        np.asarray([x for tab in tables for x in tab], dtype="<f4").tobytes(),
    ]
    header = {
        "n_trees": len(tree_sizes),
        "n_nodes": len(kinds),
        "n_internal": len(feats),
        "n_leaves": len(leaves),
        "table_sizes": [len(t) for t in tables],
        "part_bytes": [len(p) for p in parts],
        "layout": ["tree_sizes u16", "kind u8", "feature u8", "threshold_index u16", "right_offset u16", "leaf f64", "thresholds f32"],
    }
    # Pad so every part starts on an 8-byte boundary (typed-array views in JS need alignment).
    blob = b""
    offsets = []
    for p in parts:
        while len(blob) % 8:
            blob += b"\0"
        offsets.append(len(blob))
        blob += p
    header["part_offsets"] = offsets
    return blob, header


def export_psl() -> dict:
    import tldextract

    ext = tldextract.tldextract.TLD_EXTRACTOR._get_tld_extractor()
    public = sorted(ext.public_tlds)
    text = "\n".join(public)
    return {
        "source": "public suffix list as loaded by tldextract " + tldextract.__version__ + " (ICANN section; private domains excluded, as in the app)",
        "n_rules": len(public),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "rules": text,
    }


def _ranges(pred) -> list:
    out, start = [], None
    for cp in range(sys.maxunicode + 2):
        ok = cp <= sys.maxunicode and pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out += [start, cp - 1]
            start = None
    return out


def export_unicode() -> dict:
    """Python's own str predicates and lower() for non-ASCII text, so the JS port classifies
    characters exactly as the training code did (JS regex classes follow a different Unicode table)."""
    import unicodedata

    lower = {}
    for cp in range(128, sys.maxunicode + 1):
        c = chr(cp)
        if c.lower() != c:
            lower[cp] = c.lower()
    return {
        "unicode_version": unicodedata.unidata_version,
        "isspace": _ranges(str.isspace),
        "isdigit": _ranges(str.isdigit),
        "isdecimal": _ranges(str.isdecimal),
        "isalnum": _ranges(str.isalnum),
        "lower": lower,
    }


def main() -> None:
    global OUT
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="output folder (default demo/build)")
    OUT = Path(ap.parse_args().out)
    man = check_manifest()
    bundle = joblib.load(MODELS / "layer1_bundle.joblib")
    feats = list(bundle["feature_columns"])
    assert man["files"]["xgboost.joblib"]["sha256"] == man["files"]["layer1_primary.joblib"]["sha256"]
    pipes = {n: joblib.load(MODELS / f"{n}.joblib") for n in ("logistic_regression", "random_forest", "xgboost", "lightgbm")}
    for n, p in pipes.items():
        cols = list(p.named_steps["prep"].transformers_[0][2])
        assert cols == feats, f"{n} feature order differs"
    primary = bundle["pipeline"]
    cal = bundle["calibrator"]
    assert cal["type"] == "isotonic"
    iso = cal["model"]
    assert iso.out_of_bounds == "clip" and bool(iso.increasing_)

    scales = {}
    medians = {}
    for n, p in [("primary", primary), *pipes.items()]:
        scales[n], medians[n] = scaler_of(p)

    OUT.mkdir(parents=True, exist_ok=True)
    rf_blob, rf_header = export_random_forest(pipes["random_forest"])
    rf_gz = gzip.compress(rf_blob, compresslevel=9, mtime=0)
    (OUT / "model-rf.bin.gz").write_bytes(rf_gz)
    rf_header["sha256_uncompressed"] = hashlib.sha256(rf_blob).hexdigest()
    rf_header["bytes_uncompressed"] = len(rf_blob)

    core = {
        "format": 1,
        "about": "Layer-1 models of the phishing detection system, exported for the in-browser demo by demo/export_model.py.",
        "features": feats,
        "scale": scales,
        "imputer_median": medians,
        "primary": {
            "model_name": bundle["model_name"],
            "xgboost": export_xgboost(primary),
            "calibrator": {"x": [float(x) for x in iso.X_thresholds_], "y": [float(y) for y in iso.y_thresholds_]},
        },
        "witnesses": {
            "logistic_regression": export_logreg(pipes["logistic_regression"]),
            "lightgbm": export_lightgbm(pipes["lightgbm"]),
            # xgboost.joblib is byte-identical to the primary model (same sha256 in MANIFEST.json).
            "xgboost": {"same_as_primary": True},
            "random_forest": {"file": "model-rf.bin.gz", **rf_header},
        },
        "psl": export_psl(),
        "unicode": export_unicode(),
        "provenance": {
            "bundle_sha256": man["files"]["layer1_bundle.joblib"]["sha256"],
            "files_sha256": {k: v["sha256"] for k, v in man["files"].items()},
            "trained_at_commit": man.get("trained_at_commit"),
            "train_csv_sha256": man.get("train_csv_sha256"),
            "bundle_created_utc": bundle.get("created_utc"),
        },
    }
    (OUT / "model-core.json").write_text(json.dumps(core, separators=(",", ":")), encoding="utf-8")
    core_bytes = (OUT / "model-core.json").stat().st_size
    print(f"model-core.json  {core_bytes:,} bytes ({len(gzip.compress((OUT / 'model-core.json').read_bytes())):,} gzipped)")
    print(f"model-rf.bin.gz  {len(rf_gz):,} bytes ({len(rf_blob):,} uncompressed)")
    print(json.dumps({k: v for k, v in rf_header.items() if k not in ("table_sizes",)}))


if __name__ == "__main__":
    main()
