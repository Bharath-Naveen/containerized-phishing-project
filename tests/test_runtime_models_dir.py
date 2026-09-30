"""The app must not silently prefer pre-rebuild model files over the shipped verified bundle."""

import phishguard.app.ml_layer1 as ml


def test_legacy_loose_files_do_not_shadow_shipped_bundle(tmp_path, monkeypatch):
    trained = tmp_path / "outputs" / "models"
    trained.mkdir(parents=True)
    (trained / "layer1_primary.joblib").write_bytes(b"old")
    shipped = tmp_path / "models" / "layer1"
    shipped.mkdir(parents=True)
    (shipped / "layer1_bundle.joblib").write_bytes(b"new")
    monkeypatch.setattr(ml, "models_dir", lambda: trained)
    monkeypatch.setattr("phishguard.paths.project_root", lambda: tmp_path)
    assert ml.runtime_models_dir() == shipped


def test_trained_bundle_wins(tmp_path, monkeypatch):
    trained = tmp_path / "outputs" / "models"
    trained.mkdir(parents=True)
    (trained / "layer1_bundle.joblib").write_bytes(b"mine")
    monkeypatch.setattr(ml, "models_dir", lambda: trained)
    assert ml.runtime_models_dir() == trained
