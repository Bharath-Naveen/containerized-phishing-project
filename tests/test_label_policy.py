"""Label mapping invariants."""

from src.pipeline.label_policy import (
    INTERNAL_LEGIT,
    INTERNAL_PHISH,
    kaggle_status_to_internal,
)


def test_kaggle_mapping() -> None:
    assert kaggle_status_to_internal(1) == INTERNAL_LEGIT
    assert kaggle_status_to_internal(0) == INTERNAL_PHISH


def test_internal_constants() -> None:
    assert INTERNAL_LEGIT == 0
    assert INTERNAL_PHISH == 1
