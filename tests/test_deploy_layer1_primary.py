from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.pipeline.deploy_layer1_primary import deploy_latest_selected_layer1_primary


def _write_summary(base: Path, run_dir: Path) -> Path:
    summary = {
        "run_artifacts_dir": str(run_dir),
        "model_metrics": [
            {"model": "logistic_regression", "f1": 0.78, "audit_official_https_mean_phish_proba": 0.23, "roc_auc": 0.87},
            {"model": "random_forest", "f1": 0.82, "audit_official_https_mean_phish_proba": 0.20, "roc_auc": 0.91},
        ],
    }
    p = base / "fresh_data_summary.json"
    p.write_text(json.dumps(summary), encoding="utf-8")
    return p


def test_deploy_latest_selected_primary_with_backup(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "models" / "random_forest.joblib").write_bytes(b"rf")
    (run_dir / "reports" / "training_config.json").write_text(
        json.dumps({"layer1_primary_selection_policy": "composite"}),
        encoding="utf-8",
    )

    summary_path = _write_summary(tmp_path, run_dir)

    deploy_models = tmp_path / "deploy_models"
    deploy_models.mkdir(parents=True)
    current = deploy_models / "layer1_primary.joblib"
    current.write_bytes(b"old-primary")

    with patch("src.pipeline.deploy_layer1_primary.models_dir", return_value=deploy_models):
        with patch(
            "src.pipeline.deploy_layer1_primary.predict_layer1",
            return_value={"error": None, "model_path": str(current), "phish_proba": 0.5},
        ):
            out = deploy_latest_selected_layer1_primary(summary_path=summary_path)

    assert out["selected_primary_model"] == "random_forest"
    assert current.read_bytes() == b"rf"
    backup = Path(out["backup_path"])
    assert backup.is_file()
    assert backup.read_bytes() == b"old-primary"


def test_deploy_fails_when_selected_artifact_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "reports" / "training_config.json").write_text(
        json.dumps({"layer1_primary_selection_policy": "composite"}),
        encoding="utf-8",
    )
    summary_path = _write_summary(tmp_path, run_dir)

    deploy_models = tmp_path / "deploy_models"
    deploy_models.mkdir(parents=True)

    with patch("src.pipeline.deploy_layer1_primary.models_dir", return_value=deploy_models):
        try:
            deploy_latest_selected_layer1_primary(summary_path=summary_path)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "selected_primary_artifact_missing" in str(e)
