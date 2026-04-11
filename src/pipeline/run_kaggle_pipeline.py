"""End-to-end primary ML path: Kaggle CSV → clean → Layer-1 enrich → leak-safe split → train."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline.clean import clean
from src.pipeline.enrich import enrich
from src.pipeline.kaggle_ingest import ingest_kaggle
from src.pipeline.leakage_report import build_leakage_report
from src.pipeline.paths import ensure_layout, interim_dir, processed_dir, reports_dir
from src.pipeline.split_leak_safe import split_leak_safe
from src.pipeline.train import train

logger = logging.getLogger(__name__)


def run_kaggle_pipeline(
    *,
    csv_path: Path | None = None,
    download: bool = False,
    limit: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    layer1_use_dns: bool = False,
) -> None:
    ensure_layout()
    logger.info("=== kaggle_ingest ===")
    ingest_kaggle(csv_path, download=download)

    norm = interim_dir() / "kaggle_normalized.csv"
    cleaned = processed_dir() / "cleaned_kaggle.csv"
    enriched = processed_dir() / "enriched_kaggle_layer1.csv"
    tr = processed_dir() / "kaggle_train.csv"
    te = processed_dir() / "kaggle_test.csv"

    logger.info("=== clean ===")
    clean(
        norm,
        output_csv=cleaned,
        stats_json=reports_dir() / "kaggle_clean_stats.json",
    )

    logger.info("=== enrich layer1 ===")
    enrich(
        cleaned,
        limit=limit,
        resume=True,
        use_playwright=False,
        layer1_only=True,
        layer1_use_dns=layer1_use_dns,
        output_csv=enriched,
        checkpoint_name="kaggle_layer1_features.csv",
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

    # Also mirror to default names for tools that expect train.csv (optional)
    import shutil

    shutil.copy(tr, processed_dir() / "train.csv")
    shutil.copy(te, processed_dir() / "test.csv")
    logger.info("Copied kaggle_train/test -> train.csv / test.csv for app compatibility.")


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "run_kaggle_pipeline.log")
    ap = argparse.ArgumentParser(description="Primary Kaggle → Layer-1 ML pipeline.")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layer1-use-dns", action="store_true")
    args = ap.parse_args()
    run_kaggle_pipeline(
        csv_path=args.csv,
        download=args.download,
        limit=args.limit,
        test_size=args.test_size,
        seed=args.seed,
        layer1_use_dns=args.layer1_use_dns,
    )


if __name__ == "__main__":
    main()
