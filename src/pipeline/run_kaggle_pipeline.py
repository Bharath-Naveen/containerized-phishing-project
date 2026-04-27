"""End-to-end primary ML path: Kaggle CSV → clean → (optional stratified sample) → Layer-1 enrich → split → train."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.pipeline.clean import clean
from src.pipeline.enrich import enrich
from src.pipeline.fresh_dataset import (
    default_fresh_dataset_path,
    default_fresh_recent_holdout_path,
    load_and_merge_fresh_dataset,
)
from src.pipeline.kaggle_ingest import ingest_kaggle
from src.pipeline.label_policy import phish_probability_from_proba_row
from src.pipeline.leakage_report import build_leakage_report
from src.pipeline.paths import ensure_layout, interim_dir, models_dir, processed_dir, reports_dir
from src.pipeline.simple_legit_augment import augment_cleaned_with_simple_legit
from src.pipeline.split_leak_safe import split_leak_safe
from src.pipeline.stratified_sample import save_sampled_cleaned_csv, stratified_sample_by_label
from src.pipeline.train import train

logger = logging.getLogger(__name__)

# Default experiment size when not using --full (fast iteration on ~796k-row Kaggle dumps).
DEFAULT_STRATIFIED_SAMPLE_SIZE = 50_000
FRESH_SOURCE_DATASET = "fresh_phishstats_extension"


def _label_counts(df: pd.DataFrame) -> Dict[str, int]:
    if "label" not in df.columns or df.empty:
        return {}
    vc = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int).value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def _evaluate_model_on_df(model_path: Path, eval_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if eval_df.empty or "label" not in eval_df.columns:
        return None
    try:
        y_true = pd.to_numeric(eval_df["label"], errors="coerce").fillna(0).astype(int).values
        if len(y_true) == 0:
            return None
        pipe = joblib.load(model_path)
        prep = pipe.named_steps.get("prep")
        feature_cols = list(getattr(prep, "feature_names_in_", []))
        if not feature_cols:
            return None
        X = eval_df.copy()
        for c in feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[feature_cols]

        pred = pipe.predict(X)
        out: Dict[str, Any] = {
            "model": model_path.stem,
            "n_eval": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        }
        try:
            P = pipe.predict_proba(X)
            cls = np.asarray(pipe.classes_)
            phish_proba = np.array([phish_probability_from_proba_row(P[i], cls) for i in range(len(P))])
            out["roc_auc"] = (
                float(roc_auc_score(y_true, phish_proba))
                if len(np.unique(y_true)) > 1
                else None
            )
        except Exception:
            out["roc_auc"] = None
        return out
    except Exception as ex:
        logger.warning("Fresh evaluation failed for %s: %s", model_path.name, ex)
        return None


def _run_fresh_only_evaluation(
    *,
    fresh_eval_enriched_csv: Path,
) -> Optional[Dict[str, Any]]:
    if not fresh_eval_enriched_csv.is_file():
        return None
    eval_df = pd.read_csv(fresh_eval_enriched_csv, dtype=str, low_memory=False)
    if eval_df.empty:
        return None

    model_candidates = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
        "lightgbm.joblib",
        "layer1_primary.joblib",
    ]
    results: list[Dict[str, Any]] = []
    for model_name in model_candidates:
        model_path = models_dir() / model_name
        if not model_path.is_file():
            continue
        m = _evaluate_model_on_df(model_path, eval_df)
        if m is not None:
            results.append(m)
    if not results:
        return None

    out = {
        "fresh_eval_rows": int(len(eval_df)),
        "fresh_eval_label_counts": _label_counts(eval_df),
        "metrics": results,
    }
    rep_path = reports_dir() / "fresh_only_evaluation_metrics.json"
    rep_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(reports_dir() / "fresh_only_evaluation_metrics.csv", index=False)
    return out


def run_kaggle_pipeline(
    *,
    csv_path: Path | None = None,
    download: bool = False,
    limit: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    layer1_use_dns: bool = False,
    full_dataset: bool = False,
    sample_size: Optional[int] = None,
    sample_frac: Optional[float] = None,
    enrich_resume: bool = True,
    checkpoint_every: int | None = None,
    use_fresh_data: bool = False,
    fresh_dataset_csv: Path | None = None,
    fresh_recent_holdout_csv: Path | None = None,
    fresh_weight: float = 0.25,
    primary_selection: str = "composite",
    write_primary_artifact: bool = True,
) -> None:
    ensure_layout()
    logger.info("=== kaggle_ingest ===")
    ingest_kaggle(csv_path, download=download)

    norm = interim_dir() / "kaggle_normalized.csv"
    run_context: Dict[str, Any] = {
        "pipeline": "run_kaggle_pipeline",
        "use_fresh_data": bool(use_fresh_data),
        "primary_selection": primary_selection,
        "write_primary_artifact": bool(write_primary_artifact),
        "training_data_mode": "kaggle_only",
        "fresh_dataset_csv": None,
        "fresh_recent_holdout_csv": None,
        "fresh_weight": float(fresh_weight),
    }
    if use_fresh_data:
        fresh_path = fresh_dataset_csv or default_fresh_dataset_path()
        recent_path = fresh_recent_holdout_csv if fresh_recent_holdout_csv is not None else default_fresh_recent_holdout_path()
        run_context["fresh_dataset_csv"] = str(fresh_path)
        run_context["fresh_recent_holdout_csv"] = str(recent_path)
        if not fresh_path.is_file():
            logger.warning("Fresh dataset requested but missing: %s. Continuing Kaggle-only path.", fresh_path)
        else:
            logger.info("=== merge fresh dataset extension ===")
            merged_path, sanity = load_and_merge_fresh_dataset(
                kaggle_normalized_csv=norm,
                fresh_dataset_csv=fresh_path,
                fresh_recent_holdout_csv=recent_path if recent_path.is_file() else None,
            )
            norm = merged_path
            src_name = fresh_path.name.lower()
            run_context["training_data_mode"] = (
                "kaggle_plus_fresh_mock" if "mock" in src_name else "kaggle_plus_fresh_real"
            )
            run_context["fresh_merge_sanity"] = sanity
            logger.info(
                "Fresh extension applied. Combined rows=%s status_counts=%s",
                sanity.get("combined_rows_out"),
                sanity.get("status_counts_combined"),
            )
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "training_data_context_last_run.json").write_text(
        json.dumps(run_context, indent=2),
        encoding="utf-8",
    )
    cleaned_path = processed_dir() / "cleaned_kaggle.csv"
    enriched = processed_dir() / "enriched_kaggle_layer1.csv"
    tr = processed_dir() / "kaggle_train.csv"
    te = processed_dir() / "kaggle_test.csv"

    logger.info("=== clean (full dedupe) ===")
    cleaned_df = clean(
        norm,
        output_csv=cleaned_path,
        stats_json=reports_dir() / "kaggle_clean_stats.json",
    )
    n_dedup = len(cleaned_df)
    effective_cleaned_df = cleaned_df

    # Keep Kaggle as primary and cap fresh contribution to avoid dominance.
    fresh_cleaned_for_eval = pd.DataFrame()
    if use_fresh_data and "source_dataset" in cleaned_df.columns:
        source_dataset = cleaned_df["source_dataset"].fillna("").astype(str)
        fresh_mask = source_dataset.eq(FRESH_SOURCE_DATASET)
        fresh_cleaned_for_eval = cleaned_df.loc[fresh_mask].copy()
        kaggle_cleaned = cleaned_df.loc[~fresh_mask].copy()
        fresh_rows = len(fresh_cleaned_for_eval)
        kaggle_rows = len(kaggle_cleaned)
        fresh_cap = max(0, int(kaggle_rows * max(0.0, float(fresh_weight))))
        if fresh_rows > 0 and fresh_cap > 0:
            fresh_for_train, fresh_sample_stats = stratified_sample_by_label(
                fresh_cleaned_for_eval,
                n=min(fresh_rows, fresh_cap),
                random_state=seed,
            )
            effective_cleaned_df = pd.concat([kaggle_cleaned, fresh_for_train], ignore_index=True)
        else:
            fresh_for_train = fresh_cleaned_for_eval.iloc[0:0].copy()
            fresh_sample_stats = {"note": "fresh_weight produced zero fresh training rows."}
            effective_cleaned_df = kaggle_cleaned

        merge_summary = {
            "kaggle_row_count": int(kaggle_rows),
            "fresh_row_count": int(fresh_rows),
            "combined_row_count": int(len(effective_cleaned_df)),
            "class_distribution_before_merge": {
                "kaggle": _label_counts(kaggle_cleaned),
                "fresh": _label_counts(fresh_cleaned_for_eval),
            },
            "class_distribution_after_merge": _label_counts(effective_cleaned_df),
            "fresh_weight": float(fresh_weight),
            "fresh_cap_rows": int(fresh_cap),
            "fresh_rows_used_for_training": int(len(fresh_for_train)),
            "fresh_sampling_stats": fresh_sample_stats,
        }
        run_context["fresh_merge_weighted_summary"] = merge_summary
        logger.info("Weighted fresh merge summary: %s", json.dumps(merge_summary, indent=2))
        (reports_dir() / "fresh_weighted_merge_summary.json").write_text(
            json.dumps(merge_summary, indent=2),
            encoding="utf-8",
        )

    manifest: Dict[str, Any] = {
        "full_deduplicated_csv": str(cleaned_path.resolve()),
        "full_deduplicated_rows": n_dedup,
        "random_seed": seed,
        "layer1_use_dns": layer1_use_dns,
    }

    if full_dataset:
        if sample_size is not None or sample_frac is not None:
            raise ValueError("Do not pass --sample-size/--sample-frac with --full")
        full_for_enrich, aug_full = augment_cleaned_with_simple_legit(effective_cleaned_df.copy())
        manifest["simple_legit_augment"] = aug_full
        aug_csv = processed_dir() / "cleaned_kaggle_augmented_for_enrich.csv"
        full_for_enrich.to_csv(aug_csv, index=False)
        enrich_input = aug_csv
        manifest["run_mode"] = "FULL_DATASET"
        logger.info(
            "========== KAGGLE PIPELINE: FULL DEDUPLICATED DATASET (rows=%s) — expect long Layer-1 enrich ==========",
            n_dedup,
        )
        sample_rows = len(full_for_enrich)
    else:
        if sample_size is not None and sample_frac is not None:
            raise ValueError("Pass only one of --sample-size or --sample-frac")
        if sample_frac is not None:
            sampled_df, sstats = stratified_sample_by_label(
                effective_cleaned_df, frac=sample_frac, random_state=seed
            )
            manifest["sample_frac"] = sample_frac
        elif sample_size is not None:
            sampled_df, sstats = stratified_sample_by_label(
                effective_cleaned_df, n=sample_size, random_state=seed
            )
            manifest["sample_size_requested"] = sample_size
        else:
            manifest["default_sample_applied"] = DEFAULT_STRATIFIED_SAMPLE_SIZE
            sampled_df, sstats = stratified_sample_by_label(
                effective_cleaned_df, n=DEFAULT_STRATIFIED_SAMPLE_SIZE, random_state=seed
            )
        sampled_df, aug_s = augment_cleaned_with_simple_legit(sampled_df)
        manifest["simple_legit_augment"] = aug_s
        sample_rows = len(sampled_df)
        sample_path = (
            processed_dir() / f"cleaned_kaggle_sample_n{sample_rows}_rs{seed}.csv"
        )
        manifest["run_mode"] = "STRATIFIED_SAMPLE"
        manifest_path = reports_dir() / "kaggle_sample_manifest.json"
        save_sampled_cleaned_csv(
            sampled_df,
            sample_path,
            manifest={**manifest, "stratified_sample": sstats},
            manifest_path=manifest_path,
        )
        enrich_input = sample_path
        logger.info(
            "========== KAGGLE PIPELINE: STRATIFIED SAMPLE (rows=%s / dedup=%s, seed=%s) ==========",
            sample_rows,
            n_dedup,
            seed,
        )
        logger.info("Sample stats: %s", json.dumps(sstats, indent=2))

    ck_name = f"kaggle_layer1_n{sample_rows}_rs{seed}.csv"
    logger.info("=== enrich layer1 (input=%s, checkpoint=%s) ===", enrich_input, ck_name)
    enrich(
        enrich_input,
        limit=limit,
        resume=enrich_resume,
        use_playwright=False,
        layer1_only=True,
        layer1_use_dns=layer1_use_dns,
        output_csv=enriched,
        checkpoint_name=ck_name,
        checkpoint_every=checkpoint_every,
    )

    logger.info("=== split_leak_safe ===")
    split_leak_safe(
        enriched,
        test_size=test_size,
        random_state=seed,
        train_out=tr,
        test_out=te,
    )

    logger.info("=== leakage_report ===")
    build_leakage_report(tr, te)

    logger.info("=== train layer1 ===")
    train(
        tr,
        te,
        random_state=seed,
        exclude_fetch_proxy_features=True,
        layer1_only=True,
        layer1_include_dns=layer1_use_dns,
        primary_selection=primary_selection,
        write_primary_artifact=write_primary_artifact,
    )

    fresh_eval_result: Optional[Dict[str, Any]] = None
    if use_fresh_data and not fresh_cleaned_for_eval.empty:
        fresh_eval_input = processed_dir() / "fresh_eval_cleaned.csv"
        fresh_eval_enriched = processed_dir() / "fresh_eval_enriched_layer1.csv"
        fresh_cleaned_for_eval.to_csv(fresh_eval_input, index=False)
        logger.info("=== enrich layer1 (fresh-only evaluation set) ===")
        enrich(
            fresh_eval_input,
            limit=None,
            resume=enrich_resume,
            use_playwright=False,
            layer1_only=True,
            layer1_use_dns=layer1_use_dns,
            output_csv=fresh_eval_enriched,
            checkpoint_name=f"fresh_eval_layer1_rs{seed}.csv",
            checkpoint_every=checkpoint_every,
        )
        fresh_eval_result = _run_fresh_only_evaluation(fresh_eval_enriched_csv=fresh_eval_enriched)
        if fresh_eval_result is not None:
            logger.info("Fresh-only evaluation metrics written to reports.")

    standard_metrics: Optional[Any] = None
    metrics_json = reports_dir().parent / "metrics" / "metrics.json"
    if metrics_json.is_file():
        try:
            standard_metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        except Exception:
            standard_metrics = None
    eval_compare = {
        "standard_test_metrics": standard_metrics,
        "fresh_only_metrics": fresh_eval_result,
    }
    (reports_dir() / "evaluation_comparison_standard_vs_fresh.json").write_text(
        json.dumps(eval_compare, indent=2),
        encoding="utf-8",
    )

    import shutil

    shutil.copy(tr, processed_dir() / "train.csv")
    shutil.copy(te, processed_dir() / "test.csv")
    logger.info("Copied kaggle_train/test -> train.csv / test.csv for app compatibility.")


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "run_kaggle_pipeline.log")
    ap = argparse.ArgumentParser(
        description="Primary Kaggle -> Layer-1 ML pipeline (stratified sampling by default for speed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Extra cap on enrichment rows after sampling (debug).",
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sample + split (unless --random-seed set).",
    )
    ap.add_argument("--random-seed", type=int, default=None, help="Overrides --seed if provided.")
    ap.add_argument("--layer1-use-dns", action="store_true")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Enrich/train on ALL deduplicated rows (slow for ~800k URLs).",
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=f"Stratified sample count (class balance preserved). Default {DEFAULT_STRATIFIED_SAMPLE_SIZE} if not --full and no --sample-frac.",
    )
    ap.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Stratified fraction of deduplicated rows (0–1). Mutually exclusive with --sample-size.",
    )
    ap.add_argument(
        "--no-enrich-resume",
        action="store_true",
        help="Ignore Layer-1 checkpoint and recompute all features for this sample.",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Override enrich checkpoint interval (default 400 for layer1).",
    )
    ap.add_argument(
        "--use-fresh-data",
        action="store_true",
        help="Merge phishing_dataset/dataset_full.csv (fresh extension) with Kaggle normalized data.",
    )
    ap.add_argument(
        "--fresh-dataset-csv",
        type=Path,
        default=None,
        help="Path to fresh dataset_full.csv (defaults to ./phishing_dataset/dataset_full.csv).",
    )
    ap.add_argument(
        "--fresh-recent-holdout-csv",
        type=Path,
        default=None,
        help="Optional recent holdout CSV to exclude from train/test (defaults to ./phishing_dataset/dataset_test_recent.csv).",
    )
    ap.add_argument(
        "--fresh-weight",
        type=float,
        default=0.25,
        help="Maximum fresh rows as a fraction of Kaggle rows for training (0.0 disables fresh training augmentation, but fresh-only eval still runs).",
    )
    ap.add_argument(
        "--primary-selection",
        choices=["composite", "f1", "roc_auc"],
        default="composite",
        help="Policy for selecting layer1_primary.joblib (default keeps current behavior).",
    )
    ap.add_argument(
        "--no-write-primary",
        action="store_true",
        help="Run training without overwriting outputs/models/layer1_primary.joblib.",
    )
    args = ap.parse_args()
    seed = args.random_seed if args.random_seed is not None else args.seed
    if args.full and (args.sample_size is not None or args.sample_frac is not None):
        ap.error("Do not combine --full with --sample-size or --sample-frac")
    if args.sample_size is not None and args.sample_frac is not None:
        ap.error("Pass only one of --sample-size or --sample-frac")
    run_kaggle_pipeline(
        csv_path=args.csv,
        download=args.download,
        limit=args.limit,
        test_size=args.test_size,
        seed=seed,
        layer1_use_dns=args.layer1_use_dns,
        full_dataset=args.full,
        sample_size=args.sample_size,
        sample_frac=args.sample_frac,
        enrich_resume=not args.no_enrich_resume,
        checkpoint_every=args.checkpoint_every,
        use_fresh_data=args.use_fresh_data,
        fresh_dataset_csv=args.fresh_dataset_csv,
        fresh_recent_holdout_csv=args.fresh_recent_holdout_csv,
        fresh_weight=args.fresh_weight,
        primary_selection=args.primary_selection,
        write_primary_artifact=not args.no_write_primary,
    )


if __name__ == "__main__":
    main()
