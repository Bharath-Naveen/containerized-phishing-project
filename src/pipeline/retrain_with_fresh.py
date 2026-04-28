"""Retrain pipeline with optional fresh-data augmentation."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.pipeline.build_fresh_dataset import build_fresh_dataset
from src.pipeline.enrich import enrich
from src.pipeline.fresh_data import count_by, get_registered_domain, label_sanity_check
from src.pipeline.merge_datasets import merge_datasets
from src.pipeline.paths import ensure_layout, models_dir, outputs_dir, processed_dir, reports_dir
from src.pipeline.split_leak_safe import split_leak_safe
from src.pipeline.train import train

logger = logging.getLogger(__name__)
DEBUG_SAMPLE_RANDOM_STATE = 42

# Layer-1 agreement witness filenames (deployed to outputs/models/ after fresh retrain).
LAYER1_WITNESS_MODEL_BASENAMES: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
)


def persist_layer1_witness_models(run_models_dir: Path, deploy_models_dir: Path) -> list[str]:
    """
    Copy latest trained witness pipelines into ``deploy_models_dir`` (always overwrite).

    Does not touch ``layer1_primary.joblib``; that remains governed by ``overwrite_model``.
    """
    saved: list[str] = []
    deploy_models_dir.mkdir(parents=True, exist_ok=True)
    for name in LAYER1_WITNESS_MODEL_BASENAMES:
        src = run_models_dir / f"{name}.joblib"
        if not src.is_file():
            continue
        dst = deploy_models_dir / f"{name}.joblib"
        dst.write_bytes(src.read_bytes())
        logger.info("Saved model: %s", dst.name)
        saved.append(dst.name)
    return saved


def _load_kaggle_any(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if "url" not in df.columns:
        raise ValueError("Kaggle path must contain `url` column")
    if "status" in df.columns:
        df["status"] = pd.to_numeric(df["status"], errors="coerce")
    elif "label" in df.columns:
        lab = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        # internal label: 1=phish,0=legit -> status: 0=phish,1=legit
        df["status"] = lab.map(lambda x: 0 if int(x) == 1 else 1)
    else:
        raise ValueError("Kaggle path must include either `status` or `label`")
    df["status"] = pd.to_numeric(df["status"], errors="coerce")
    df = label_sanity_check(df)
    df["label"] = df["status"].map(lambda s: 1 if int(s) == 0 else 0).astype(int)
    if "registered_domain" not in df.columns:
        df["registered_domain"] = df["url"].map(get_registered_domain)
    df["source"] = df.get("source", "kaggle")
    return df


def _set_outputs_env(temp_outputs: Path) -> Dict[str, Optional[str]]:
    prev = {"PHISH_OUTPUTS_DIR": os.environ.get("PHISH_OUTPUTS_DIR")}
    os.environ["PHISH_OUTPUTS_DIR"] = str(temp_outputs)
    return prev


def _restore_env(prev: Dict[str, Optional[str]]) -> None:
    old = prev.get("PHISH_OUTPUTS_DIR")
    if old is None:
        os.environ.pop("PHISH_OUTPUTS_DIR", None)
    else:
        os.environ["PHISH_OUTPUTS_DIR"] = old


def ensure_split_compatible(enriched_csv: Path) -> Path:
    df = pd.read_csv(enriched_csv, dtype=str, low_memory=False)
    changed = False

    if "canonical_url" not in df.columns:
        if "url" in df.columns:
            df["canonical_url"] = df["url"].astype(str).str.strip()
            changed = True
        elif "input_url" in df.columns:
            df["canonical_url"] = df["input_url"].astype(str).str.strip()
            changed = True
        else:
            raise ValueError("Split compatibility failed: missing canonical_url and no url/input_url fallback.")

    if "label" not in df.columns:
        if "status" not in df.columns:
            raise ValueError("Split compatibility failed: missing both label and status columns.")
        st = pd.to_numeric(df["status"], errors="coerce")
        df["label"] = st.map(lambda s: 1 if int(s) == 0 else 0 if pd.notna(s) else pd.NA)
        changed = True

    if "status" not in df.columns:
        lb = pd.to_numeric(df["label"], errors="coerce")
        df["status"] = lb.map(lambda s: 0 if int(s) == 1 else 1 if pd.notna(s) else pd.NA)
        changed = True

    df = label_sanity_check(df)
    if changed:
        out_path = enriched_csv.with_name(f"{enriched_csv.stem}_split_compatible{enriched_csv.suffix}")
        df.to_csv(out_path, index=False)
        logger.info("Wrote split-compatible enriched CSV -> %s", out_path)
        return out_path
    return enriched_csv


def ensure_enrich_compatible(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "canonical_url" not in out.columns:
        if "url" in out.columns:
            out["canonical_url"] = out["url"].astype(str).str.strip()
        elif "input_url" in out.columns:
            out["canonical_url"] = out["input_url"].astype(str).str.strip()
        else:
            raise ValueError("Enrich compatibility failed: missing canonical_url and no url/input_url fallback.")
    return out


def _sample_merged_rows_balanced(df: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
    if sample_rows <= 0 or df.empty or len(df) <= sample_rows:
        return df
    if "status" not in df.columns:
        return df.sample(n=sample_rows, random_state=DEBUG_SAMPLE_RANDOM_STATE).reset_index(drop=True)

    d = df.copy()
    d["status"] = pd.to_numeric(d["status"], errors="coerce")
    d = d[d["status"].isin([0, 1])].copy()
    if d.empty:
        return df.sample(n=min(sample_rows, len(df)), random_state=DEBUG_SAMPLE_RANDOM_STATE).reset_index(drop=True)

    counts = d["status"].value_counts()
    if len(counts) < 2:
        return d.sample(n=min(sample_rows, len(d)), random_state=DEBUG_SAMPLE_RANDOM_STATE).reset_index(drop=True)

    target0 = int(round(sample_rows * (counts.get(0, 0) / len(d))))
    target1 = sample_rows - target0
    s0 = d[d["status"] == 0].sample(n=min(target0, int(counts.get(0, 0))), random_state=DEBUG_SAMPLE_RANDOM_STATE)
    s1 = d[d["status"] == 1].sample(n=min(target1, int(counts.get(1, 0))), random_state=DEBUG_SAMPLE_RANDOM_STATE)
    out = pd.concat([s0, s1], ignore_index=True)
    if len(out) < sample_rows:
        remaining = d.loc[~d.index.isin(out.index)]
        needed = min(sample_rows - len(out), len(remaining))
        if needed > 0:
            out = pd.concat(
                [out, remaining.sample(n=needed, random_state=DEBUG_SAMPLE_RANDOM_STATE)],
                ignore_index=True,
            )
    return out.sample(frac=1.0, random_state=DEBUG_SAMPLE_RANDOM_STATE).reset_index(drop=True)


def _sample_with_fresh_preserved(df: pd.DataFrame, sample_rows: int) -> tuple[pd.DataFrame, int, int]:
    if sample_rows <= 0 or df.empty or len(df) <= sample_rows:
        fresh_rows = int((df.get("source", pd.Series([""] * len(df))).astype(str).str.lower() != "kaggle").sum())
        kaggle_rows = int(len(df) - fresh_rows)
        return df, fresh_rows, kaggle_rows
    if "source" not in df.columns:
        sampled = _sample_merged_rows_balanced(df, sample_rows)
        return sampled, 0, int(len(sampled))

    d = df.copy()
    src = d["source"].fillna("").astype(str).str.lower()
    fresh = d[src != "kaggle"].copy()
    kag = d[src == "kaggle"].copy()

    if len(fresh) >= sample_rows:
        sampled = _sample_merged_rows_balanced(fresh, sample_rows)
        return sampled, int(len(sampled)), 0

    remaining = sample_rows - len(fresh)
    kag_sampled = _sample_merged_rows_balanced(kag, remaining)
    sampled = pd.concat([fresh, kag_sampled], ignore_index=True)
    sampled = sampled.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    sampled = sampled.sample(frac=1.0, random_state=DEBUG_SAMPLE_RANDOM_STATE).reset_index(drop=True)
    fresh_rows = int((sampled["source"].fillna("").astype(str).str.lower() != "kaggle").sum())
    kaggle_rows = int(len(sampled) - fresh_rows)
    return sampled, fresh_rows, kaggle_rows


def retrain_with_fresh(
    *,
    kaggle_path: Path,
    use_fresh_data: bool = False,
    fresh_weight: float = 0.3,
    n_phishing: int = 2000,
    n_legitimate: int = 2000,
    overwrite_model: bool = False,
    sample_rows: Optional[int] = None,
    fresh_preserve_in_sample: bool = False,
) -> Dict[str, Any]:
    ensure_layout()
    kag = _load_kaggle_any(kaggle_path)
    fresh_train = pd.DataFrame(columns=kag.columns)
    fresh_holdout = pd.DataFrame(columns=kag.columns)
    fresh_collection_meta: Dict[str, Any] = {
        "phishstats_rows_collected": 0,
        "phishstats_errors_count": 0,
        "tranco_rows_collected": 0,
        "tranco_download_failed": False,
        "tranco_error": None,
    }
    if use_fresh_data:
        phish_pages = 3 if (sample_rows is not None and int(sample_rows) > 0) else 12
        fresh_train, fresh_holdout, fresh_collection_meta = build_fresh_dataset(
            n_phishing=n_phishing,
            n_legitimate=n_legitimate,
            phish_pages=phish_pages,
            phish_timeout_s=20,
            phish_max_failed_pages=3,
        )
        if not fresh_train.empty:
            fresh_train["label"] = fresh_train["status"].map(lambda s: 1 if int(s) == 0 else 0).astype(int)
            fresh_train["source"] = fresh_train.get("source", "fresh")
            for c in kag.columns:
                if c not in fresh_train.columns:
                    fresh_train[c] = pd.NA
            fresh_train = fresh_train[kag.columns]
    merged, merge_stats = merge_datasets(
        kag,
        fresh_train,
        fresh_holdout_df=fresh_holdout,
        fresh_weight=fresh_weight,
        return_stats=True,
    )
    fresh_collection_meta["fresh_vs_kaggle_domain_overlap_removed"] = int(
        merge_stats.get("fresh_vs_kaggle_domain_overlap_removed", 0)
    )
    fresh_collection_meta["fresh_vs_holdout_domain_overlap_removed"] = int(
        merge_stats.get("fresh_vs_holdout_domain_overlap_removed", 0)
    )
    sampled_debug_mode = bool(sample_rows is not None and int(sample_rows) > 0)
    pre_sample_rows = int(len(merged))
    fresh_rows_in_sample = int((merged.get("source", pd.Series([""] * len(merged))).astype(str).str.lower() != "kaggle").sum())
    kaggle_rows_in_sample = int(len(merged) - fresh_rows_in_sample)
    if sampled_debug_mode:
        if use_fresh_data and fresh_preserve_in_sample:
            merged, fresh_rows_in_sample, kaggle_rows_in_sample = _sample_with_fresh_preserved(merged, int(sample_rows))
        else:
            merged = _sample_merged_rows_balanced(merged, int(sample_rows))
            fresh_rows_in_sample = int(
                (merged.get("source", pd.Series([""] * len(merged))).astype(str).str.lower() != "kaggle").sum()
            )
            kaggle_rows_in_sample = int(len(merged) - fresh_rows_in_sample)
        logger.info(
            "Debug sample mode enabled: sampled_rows=%s from merged_rows=%s (random_state=%s)",
            len(merged),
            pre_sample_rows,
            DEBUG_SAMPLE_RANDOM_STATE,
        )

    merged = ensure_enrich_compatible(merged)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    work = processed_dir()
    merged_csv = work / f"retrain_with_fresh_merged_{ts}.csv"
    merged.to_csv(merged_csv, index=False)
    enriched_csv = work / f"retrain_with_fresh_enriched_{ts}.csv"
    enrich(
        merged_csv,
        layer1_only=True,
        use_playwright=False,
        layer1_use_dns=False,
        output_csv=enriched_csv,
        resume=False,
        checkpoint_name=f"retrain_with_fresh_ckpt_{ts}.csv",
    )
    train_csv = work / f"retrain_with_fresh_train_{ts}.csv"
    test_csv = work / f"retrain_with_fresh_test_{ts}.csv"
    split_input_csv = ensure_split_compatible(enriched_csv)
    split_leak_safe(split_input_csv, train_out=train_csv, test_out=test_csv)
    split_stats_path = reports_dir() / "split_leak_safe_stats.json"
    split_stats: Dict[str, Any] = {}
    if split_stats_path.is_file():
        try:
            split_stats = json.loads(split_stats_path.read_text(encoding="utf-8"))
        except Exception:
            split_stats = {}

    temp_outputs = outputs_dir() / "fresh_retrain_runs" / ts
    temp_outputs.mkdir(parents=True, exist_ok=True)
    prev_env = _set_outputs_env(temp_outputs)
    try:
        train(
            train_csv,
            test_csv,
            layer1_only=True,
            layer1_include_dns=False,
            primary_selection="composite",
            write_primary_artifact=False,
        )
    finally:
        _restore_env(prev_env)

    run_models_dir = temp_outputs / "models"
    witness_models_saved = persist_layer1_witness_models(run_models_dir, models_dir())
    run_metrics_path = temp_outputs / "metrics" / "metrics.json"
    metrics = json.loads(run_metrics_path.read_text(encoding="utf-8")) if run_metrics_path.is_file() else []
    best = max(metrics, key=lambda m: float(m.get("f1") or 0.0), default=None)
    model_saved: Optional[str] = None
    if best:
        src = run_models_dir / f"{best['model']}.joblib"
        if src.is_file():
            if overwrite_model:
                dst = models_dir() / "layer1_primary.joblib"
            else:
                dst = models_dir() / f"layer1_primary_{ts}.joblib"
            dst.write_bytes(src.read_bytes())
            model_saved = str(dst)

    # Optional xgboost availability marker
    xgb_available = bool((run_models_dir / "xgboost.joblib").is_file())
    leakage: Dict[str, Any] = {}
    try:
        leakage["train_test_domain_overlap"] = int(split_stats.get("registered_domain_overlap_count", 0))
        leakage["train_test_url_overlap"] = int(split_stats.get("canonical_url_overlap_count", 0))
        leakage["fresh_vs_kaggle_domain_overlap_removed"] = int(
            fresh_collection_meta.get("fresh_vs_kaggle_domain_overlap_removed", 0)
        )
        leakage["fresh_vs_holdout_domain_overlap_removed"] = int(
            fresh_collection_meta.get("fresh_vs_holdout_domain_overlap_removed", 0)
        )
    except Exception:
        leakage["train_test_domain_overlap"] = None
        leakage["train_test_url_overlap"] = None

    fresh_rows_used = int(merge_stats["fresh_rows_used"])
    fresh_effective = bool(use_fresh_data and fresh_rows_used > 0)
    warnings: list[str] = []
    if use_fresh_data and fresh_rows_used == 0:
        warnings.append("Fresh data requested but no fresh rows were used in training.")
    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "kaggle_path": str(kaggle_path),
        "use_fresh_data": bool(use_fresh_data),
        "fresh_collection_attempted": bool(use_fresh_data),
        "phishstats_rows_collected": int(fresh_collection_meta.get("phishstats_rows_collected", 0)),
        "tranco_rows_collected": int(fresh_collection_meta.get("tranco_rows_collected", 0)),
        "phishstats_errors_count": int(fresh_collection_meta.get("phishstats_errors_count", 0)),
        "tranco_download_failed": bool(fresh_collection_meta.get("tranco_download_failed", False)),
        "tranco_error": fresh_collection_meta.get("tranco_error"),
        "fresh_data_effective": fresh_effective,
        "warnings": warnings,
        "fresh_weight": float(fresh_weight),
        "n_phishing_requested": int(n_phishing),
        "n_legitimate_requested": int(n_legitimate),
        "debug_sample_mode": sampled_debug_mode,
        "sample_rows_requested": int(sample_rows) if sample_rows is not None else None,
        "sample_random_state": DEBUG_SAMPLE_RANDOM_STATE if sampled_debug_mode else None,
        "fresh_preserve_in_sample": bool(fresh_preserve_in_sample),
        "fresh_rows_in_sample": int(fresh_rows_in_sample),
        "kaggle_rows_in_sample": int(kaggle_rows_in_sample),
        "fresh_sample_fraction": (float(fresh_rows_in_sample) / float(len(merged))) if len(merged) > 0 else 0.0,
        "dataset_size": {
            "kaggle_rows": int(merge_stats["kaggle_rows"]),
            "fresh_rows_available": int(merge_stats["fresh_rows_available"]),
            "fresh_rows_used": int(merge_stats["fresh_rows_used"]),
            "fresh_train_rows": int(len(fresh_train)),
            "fresh_holdout_rows": int(len(fresh_holdout)),
            "merged_rows_before_sampling": int(pre_sample_rows),
            "merged_rows": int(len(merged)),
        },
        "class_balance_status": {
            "kaggle": kag["status"].value_counts().to_dict(),
            "merged": merged["status"].value_counts().to_dict(),
        },
        "source_counts": count_by(merged, ["source", "status"]),
        "leakage_counts": leakage,
        "model_metrics": metrics,
        "xgboost_available": xgb_available,
        "model_saved_path": model_saved,
        "overwrite_model": bool(overwrite_model),
        "witness_models_saved": witness_models_saved,
        "run_artifacts_dir": str(temp_outputs),
    }
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "fresh_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Fresh retraining summary -> %s", reports_dir() / "fresh_data_summary.json")
    return summary


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "retrain_with_fresh.log")
    ap = argparse.ArgumentParser(description="Retrain with optional fresh-data augmentation.")
    ap.add_argument("--kaggle-path", type=Path, required=True)
    ap.add_argument("--use-fresh-data", action="store_true")
    ap.add_argument("--fresh-weight", type=float, default=0.3)
    ap.add_argument("--n-phishing", type=int, default=2000)
    ap.add_argument("--n-legitimate", type=int, default=2000)
    ap.add_argument("--sample-rows", type=int, default=None)
    ap.add_argument("--fresh-preserve-in-sample", action="store_true")
    ap.add_argument("--overwrite-model", action="store_true")
    args = ap.parse_args()
    out = retrain_with_fresh(
        kaggle_path=args.kaggle_path,
        use_fresh_data=args.use_fresh_data,
        fresh_weight=args.fresh_weight,
        n_phishing=args.n_phishing,
        n_legitimate=args.n_legitimate,
        sample_rows=args.sample_rows,
        fresh_preserve_in_sample=args.fresh_preserve_in_sample,
        overwrite_model=args.overwrite_model,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
