from pathlib import Path
from unittest.mock import patch

from src.app_v1.utils.download_models import ensure_fasttext_model, resolve_fasttext_model_path


def test_resolve_fasttext_model_path_falls_back_to_repo_models() -> None:
    with patch.dict("os.environ", {}, clear=False):
        p = resolve_fasttext_model_path()
    assert str(p).endswith(str(Path("models") / "lid.176.ftz"))


def test_ensure_fasttext_model_missing_is_graceful_without_download() -> None:
    with patch("src.app_v1.utils.download_models.resolve_fasttext_model_path", return_value=Path("missing-model.ftz")):
        ok, model_path, err = ensure_fasttext_model(trigger_download=False)
    assert ok is False
    assert model_path is None
    assert err is not None and "fasttext_model_not_found" in err
