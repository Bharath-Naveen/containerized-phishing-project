"""Optional smoke: load Layer-1 model if present."""

import pytest

from src.pipeline.paths import models_dir, reports_dir


def test_layer1_model_predict_if_exists() -> None:
    model = models_dir() / "layer1_primary.joblib"
    cfg = reports_dir() / "training_config.json"
    if not model.is_file() or not cfg.is_file():
        pytest.skip("No trained model or training_config (run pipeline first)")

    from src.app_v1.ml_layer1 import predict_layer1

    out = predict_layer1("https://example.com/path", model_path=model)
    assert out.get("phish_proba") is not None
    assert 0.0 <= float(out["phish_proba"]) <= 1.0
