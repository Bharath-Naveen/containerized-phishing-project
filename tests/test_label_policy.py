"""Label mapping invariants."""

import numpy as np
import pytest

from src.pipeline.label_policy import (
    INTERNAL_LEGIT,
    INTERNAL_PHISH,
    class_index_for_internal_label,
    kaggle_status_to_internal,
    phish_probability_from_proba_row,
)


def test_kaggle_mapping() -> None:
    assert kaggle_status_to_internal(1) == INTERNAL_LEGIT
    assert kaggle_status_to_internal(0) == INTERNAL_PHISH


def test_internal_constants() -> None:
    assert INTERNAL_LEGIT == 0
    assert INTERNAL_PHISH == 1


def test_class_index_for_internal_label_sorted_sklearn_order() -> None:
    classes = np.array([0, 1])
    assert class_index_for_internal_label(classes, INTERNAL_PHISH) == 1
    assert class_index_for_internal_label(classes, INTERNAL_LEGIT) == 0


def test_phish_probability_uses_classes_not_column_assumption() -> None:
    """If ``classes_`` were ever [1, 0], column index for phishing must follow ``classes_``."""
    classes = np.array([1, 0])
    proba_row = np.array([0.2, 0.8])
    assert phish_probability_from_proba_row(proba_row, classes) == pytest.approx(0.2)


def test_phish_probability_standard_order() -> None:
    classes = np.array([0, 1])
    proba_row = np.array([0.91, 0.09])
    assert phish_probability_from_proba_row(proba_row, classes) == pytest.approx(0.09)


def test_class_index_rejects_missing_label() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        class_index_for_internal_label(np.array([0]), INTERNAL_PHISH)
