"""End-to-end primary ML path: Kaggle CSV → clean → (optional stratified sample) → Layer-1 enrich → split → train."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.pipeline.clean import clean
from src.pipeline.enrich import enrich
from src.pipeline.kaggle_ingest import ingest_kaggle
from src.pipeline.leakage_report import build_leakage_report
from src.pipeline.paths import ensure_layout, interim_dir, processed_dir, reports_dir
from src.pipeline.simple_legit_augment import augment_cleaned_with_simple_legit
from src.pipeline.split_leak_safe import split_leak_safe
from src.pipeline.stratified_sample import save_sampled_cleaned_csv, stratified_sample_by_label
from src.pipeline.train import train

logger = logging.getLogger(__name__)

# Default experiment size when not using --full (fast iteration on ~796k-row Kaggle dumps).
DEFAULT_STRATIFIED_SAMPLE_SIZE = 50_000


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
) -> None:
    ensure_layout()
    logger.info("=== kaggle_ingest ===")
    ingest_kaggle(csv_path, download=download)

    norm = interim_dir() / "kaggle_normalized.csv"
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

    manifest: Dict[str, Any] = {
        "full_deduplicated_csv": str(cleaned_path.resolve()),
        "full_deduplicated_rows": n_dedup,
        "random_seed": seed,
        "layer1_use_dns": layer1_use_dns,
    }

    if full_dataset:
        if sample_size is not None or sample_frac is not None:
            raise ValueError("Do not pass --sample-size/--sample-frac with --full")
        full_for_enrich, aug_full = augment_cleaned_with_simple_legit(cleaned_df.copy())
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
                cleaned_df, frac=sample_frac, random_state=seed
            )
            manifest["sample_frac"] = sample_frac
        elif sample_size is not None:
            sampled_df, sstats = stratified_sample_by_label(
                cleaned_df, n=sample_size, random_state=seed
            )
            manifest["sample_size_requested"] = sample_size
        else:
            manifest["default_sample_applied"] = DEFAULT_STRATIFIED_SAMPLE_SIZE
            sampled_df, sstats = stratified_sample_by_label(
                cleaned_df, n=DEFAULT_STRATIFIED_SAMPLE_SIZE, random_state=seed
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
        description="Primary Kaggle → Layer-1 ML pipeline (stratified sampling by default for speed).",
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
    )


if __name__ == "__main__":
    main()
