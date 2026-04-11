"""Run ingest → clean → enrich → analyze → prepare_ml → split → train."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline.analyze_dataset import analyze
from src.pipeline.clean import clean
from src.pipeline.dataset_report import build_full_report
from src.pipeline.enrich import enrich
from src.pipeline.ingest import ingest
from src.pipeline.logging_util import setup_logging
from src.pipeline.paths import ensure_layout, logs_dir, processed_dir
from src.pipeline.prepare_ml_dataset import prepare_ml_dataset
from src.pipeline.split import split_dataset
from src.pipeline.train import train

logger = logging.getLogger(__name__)


def main() -> None:
    ensure_layout()
    setup_logging(logs_dir() / "run_all.log")
    p = argparse.ArgumentParser(description="Full ML pipeline (run inside Docker for scraping).")
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Limit rows for enrich (debug)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--playwright", action="store_true", help="Passive Playwright behavior probe")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--brand-holdout", type=str, default=None)
    p.add_argument(
        "--skip-ml-prepare",
        action="store_true",
        help="Skip analyze + row filtering; split reads enriched.csv only.",
    )
    p.add_argument(
        "--max-phish-fetch-fail-frac",
        type=float,
        default=0.28,
        help="Cap fraction of phishing rows that may have failed HTTP fetch (after prepare).",
    )
    p.add_argument(
        "--train-phish-legit-ratio",
        type=float,
        default=None,
        help="After grouped split, cap train phishing at this × train legit count (try 1.0 or 2.0 for balanced experiments).",
    )
    p.add_argument(
        "--train-balance-rowwise",
        action="store_true",
        help="Phish downsampling without group-aware packing.",
    )
    args = p.parse_args()

    logger.info("=== ingest ===")
    ingest(args.raw_dir)
    logger.info("=== clean ===")
    clean()
    logger.info("=== enrich ===")
    enrich(limit=args.limit, resume=not args.no_resume, use_playwright=args.playwright)
    if not args.skip_ml_prepare:
        logger.info("=== analyze_dataset ===")
        analyze()
        logger.info("=== prepare_ml_dataset ===")
        prepare_ml_dataset(max_phish_fetch_fail_fraction=args.max_phish_fetch_fail_frac)
    logger.info("=== split ===")
    split_input = processed_dir() / "enriched.csv" if args.skip_ml_prepare else None
    split_dataset(
        enriched_csv=split_input,
        test_size=args.test_size,
        brand_holdout=args.brand_holdout,
        train_phish_legit_ratio=args.train_phish_legit_ratio,
        train_balance_group_aware=not args.train_balance_rowwise,
    )
    logger.info("=== train ===")
    train()
    logger.info("=== dataset_report ===")
    build_full_report()


if __name__ == "__main__":
    main()
