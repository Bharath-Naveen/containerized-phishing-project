from pathlib import Path
from unittest.mock import patch

import src.app_v1.analyze_dashboard as analyze_dashboard
from src.app_v1.analyze_dashboard import _fasttext_language_enrichment


def test_language_enrichment_graceful_without_text() -> None:
    out = _fasttext_language_enrichment({"visible_text": ""}, soup=None)
    assert out["detected_language"] is None
    assert out["detected_language_confidence"] is None
    assert out["language_detection_available"] is False
    assert out["language_mismatch_contextual_signal"] is False


def test_language_enrichment_short_text_is_contextual_na() -> None:
    out = _fasttext_language_enrichment({"visible_text": "hello world"}, soup=None)
    assert out["language_detection_available"] is False
    assert out["language_detection_error"] == "insufficient_text"


def test_language_enrichment_missing_model_path_fails_gracefully() -> None:
    analyze_dashboard._FASTTEXT_MODEL_CACHE = None
    analyze_dashboard._FASTTEXT_MODEL_ERROR = None
    with patch("src.app_v1.analyze_dashboard.resolve_fasttext_model_path", return_value=Path("not-here.ftz")):
        out = _fasttext_language_enrichment({"visible_text": "x" * 200}, soup=None)
    assert out["language_detection_available"] is False
    assert str(out["language_detection_error"]).startswith(
        ("fasttext_model_not_found", "fasttext_import_error")
    )
