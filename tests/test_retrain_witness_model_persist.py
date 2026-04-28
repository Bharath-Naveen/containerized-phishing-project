"""Tests for persisting Layer-1 witness models after retrain_with_fresh."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.retrain_with_fresh import LAYER1_WITNESS_MODEL_BASENAMES, persist_layer1_witness_models


def _tiny_binary_pipeline() -> Pipeline:
    X = pd.DataFrame({"f0": [0.0, 1.0, 0.5, 0.2], "f1": [1.0, 0.0, 0.5, 0.8]})
    y = np.array([0, 1, 1, 0])
    pipe = Pipeline(
        [
            ("prep", StandardScaler()),
            ("model", LogisticRegression(max_iter=200, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def test_persist_layer1_witness_models_creates_files_and_predict_proba(tmp_path: Path) -> None:
    run_models = tmp_path / "run" / "models"
    run_models.mkdir(parents=True)
    deploy = tmp_path / "deploy" / "models"
    pipe = _tiny_binary_pipeline()
    joblib.dump(pipe, run_models / "logistic_regression.joblib")
    joblib.dump(pipe, run_models / "random_forest.joblib")

    saved = persist_layer1_witness_models(run_models, deploy)

    assert set(saved) >= {"logistic_regression.joblib", "random_forest.joblib"}
    for name in ("logistic_regression", "random_forest"):
        dst = deploy / f"{name}.joblib"
        assert dst.is_file()
        loaded = joblib.load(dst)
        Xp = pd.DataFrame({"f0": [0.3], "f1": [0.7]})
        proba = loaded.predict_proba(Xp)
        assert proba.shape == (1, 2)


def test_persist_skips_missing_optional_models(tmp_path: Path) -> None:
    run_models = tmp_path / "run" / "models"
    run_models.mkdir(parents=True)
    deploy = tmp_path / "deploy" / "models"
    pipe = _tiny_binary_pipeline()
    joblib.dump(pipe, run_models / "logistic_regression.joblib")

    saved = persist_layer1_witness_models(run_models, deploy)

    assert saved == ["logistic_regression.joblib"]
    assert not (deploy / "xgboost.joblib").is_file()
    assert LAYER1_WITNESS_MODEL_BASENAMES == (
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    )


def test_persist_overwrites_existing(tmp_path: Path) -> None:
    run_models = tmp_path / "run" / "models"
    run_models.mkdir(parents=True)
    deploy = tmp_path / "deploy" / "models"
    deploy.mkdir(parents=True)
    (deploy / "logistic_regression.joblib").write_bytes(b"old")

    pipe = _tiny_binary_pipeline()
    joblib.dump(pipe, run_models / "logistic_regression.joblib")

    persist_layer1_witness_models(run_models, deploy)

    loaded = joblib.load(deploy / "logistic_regression.joblib")
    assert hasattr(loaded, "predict_proba")
