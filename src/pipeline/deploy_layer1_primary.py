"""Safely deploy latest selected Layer-1 primary artifact from fresh retrain runs."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from src.app_v1.ml_layer1 import predict_layer1
from src.pipeline.paths import models_dir, reports_dir
from src.pipeline.train import _pick_layer1_primary_model_by_policy

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_file:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_latest_run_info(summary_path: Path) -> tuple[Path, list[dict[str, Any]], str]:
    summary = _load_json(summary_path)
    run_dir_s = str(summary.get("run_artifacts_dir") or "").strip()
    if not run_dir_s:
        raise ValueError("fresh_data_summary.json missing run_artifacts_dir")
    run_dir = Path(run_dir_s)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_artifacts_dir_not_found:{run_dir}")

    metrics = summary.get("model_metrics")
    if not isinstance(metrics, list) or not metrics:
        mpath = run_dir / "metrics" / "metrics.json"
        metrics = _load_json(mpath)
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("no model_metrics in summary or run metrics.json")

    policy = "composite"
    cfg_path = run_dir / "reports" / "training_config.json"
    if cfg_path.is_file():
        cfg = _load_json(cfg_path)
        p = str(cfg.get("layer1_primary_selection_policy") or "").strip().lower()
        if p in {"composite", "f1", "roc_auc"}:
            policy = p
    return run_dir, metrics, policy


def deploy_latest_selected_layer1_primary(*, summary_path: Path | None = None, verify_url: str = "https://example.com") -> Dict[str, str]:
    sp = summary_path or (reports_dir() / "fresh_data_summary.json")
    run_dir, metrics, policy = _resolve_latest_run_info(sp)
    selected = _pick_layer1_primary_model_by_policy(metrics, policy=policy)

    src = run_dir / "models" / f"{selected}.joblib"
    if not src.is_file():
        raise FileNotFoundError(f"selected_primary_artifact_missing:{src}")

    md = models_dir()
    md.mkdir(parents=True, exist_ok=True)
    dst = md / "layer1_primary.joblib"

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_path = md / f"layer1_primary_backup_{ts}.joblib"
    if dst.is_file():
        backup_path.write_bytes(dst.read_bytes())
    else:
        backup_path = Path("")

    dst.write_bytes(src.read_bytes())

    out = predict_layer1(verify_url, model_path=dst)
    if str(out.get("error") or "").startswith("no_model_file"):
        raise RuntimeError("predict_layer1 could not load deployed layer1_primary.joblib")

    return {
        "backup_path": str(backup_path) if str(backup_path) else "none",
        "deployed_source": str(src),
        "deployed_target": str(dst),
        "selected_primary_model": str(selected),
        "selection_policy": str(policy),
    }


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "deploy_layer1_primary.log")
    ap = argparse.ArgumentParser(description="Safely deploy latest selected Layer-1 primary model.")
    ap.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional fresh_data_summary.json path (defaults to outputs/reports/fresh_data_summary.json).",
    )
    ap.add_argument("--verify-url", type=str, default="https://example.com")
    args = ap.parse_args()

    out = deploy_latest_selected_layer1_primary(summary_path=args.summary_path, verify_url=args.verify_url)
    print(
        "Primary deployment complete. "
        f"backup={out['backup_path']} ; source={out['deployed_source']} ; target={out['deployed_target']}"
    )


if __name__ == "__main__":
    main()
