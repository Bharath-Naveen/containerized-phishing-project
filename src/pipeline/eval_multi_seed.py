"""Optional repeated evaluation: re-split + train with different seeds (slow)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.pipeline.leakage_report import build_leakage_report
from src.pipeline.paths import ensure_layout, processed_dir, reports_dir
from src.pipeline.split_leak_safe import split_leak_safe
from src.pipeline.train import train

logger = logging.getLogger(__name__)


def eval_multi_seed(
    enriched_csv: Path,
    *,
    seeds: List[int],
    test_size: float = 0.2,
    layer1_include_dns: bool = False,
) -> Path:
    ensure_layout()
    results: List[Dict[str, Any]] = []
    tr = processed_dir() / "kaggle_train.csv"
    te = processed_dir() / "kaggle_test.csv"
    for s in seeds:
        logger.info("=== seed %s ===", s)
        split_leak_safe(enriched_csv, test_size=test_size, random_state=s, train_out=tr, test_out=te)
        build_leakage_report(tr, te)
        train(
            tr,
            te,
            random_state=s,
            layer1_only=True,
            layer1_include_dns=layer1_include_dns,
        )
        mp = reports_dir().parent / "metrics" / "metrics.json"
        if mp.is_file():
            results.append({"seed": s, "metrics": json.loads(mp.read_text(encoding="utf-8"))})
    out = reports_dir() / "multi_seed_eval.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "eval_multi_seed.log")
    ap = argparse.ArgumentParser(description="Multi-seed split+train on fixed enriched CSV.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--layer1-include-dns", action="store_true")
    args = ap.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    p = eval_multi_seed(
        args.input,
        seeds=seeds,
        test_size=args.test_size,
        layer1_include_dns=args.layer1_include_dns,
    )
    print("Wrote", p)


if __name__ == "__main__":
    main()
