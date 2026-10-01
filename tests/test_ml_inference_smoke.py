"""Smoke: the app's Layer-1 model (trained in outputs/models, or the verified one shipped in models/layer1) scores a URL."""

import pytest

from phishguard.app.ml_layer1 import load_layer1_bundle, predict_layer1, runtime_models_dir


def test_layer1_model_predicts() -> None:
    if load_layer1_bundle() is None and not (runtime_models_dir() / "layer1_primary.joblib").is_file():
        pytest.skip("No model available")
    out = predict_layer1("https://example.com/path")
    assert out.get("phish_proba") is not None
    assert 0.0 <= float(out["phish_proba"]) <= 1.0


def test_shipped_model_manifest_matches_files() -> None:
    import hashlib
    import json

    from phishguard.paths import project_root

    d = project_root() / "models" / "layer1"
    if not (d / "MANIFEST.json").is_file():
        pytest.skip("no shipped model")
    man = json.loads((d / "MANIFEST.json").read_text())
    for name, meta in man["files"].items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == meta["sha256"], name
