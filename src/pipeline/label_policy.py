"""Binary label convention for this project (single source of truth).

Internal convention (used in ``label`` column for training/evaluation):
  - **0** = legitimate
  - **1** = phishing

Many public datasets (including the Kaggle set referenced in docs) encode:
  - **1** = legitimate
  - **0** = phishing

Use :func:`kaggle_status_to_internal` when ingesting those sources.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

INTERNAL_LEGIT = 0
INTERNAL_PHISH = 1


def class_index_for_internal_label(
    classes: Union[np.ndarray, Sequence],
    internal_label: int,
) -> int:
    """
    Index into ``predict_proba`` row for a given internal label value.

    Training uses :data:`INTERNAL_LEGIT` / :data:`INTERNAL_PHISH`; sklearn's ``classes_``
    lists those values in sorted order — do not assume phishing is always column 1.
    """
    arr = np.asarray(classes)
    matches = np.flatnonzero(arr == internal_label)
    if matches.size != 1:
        raise ValueError(
            f"Expected exactly one entry equal to {internal_label} in classes_={arr!r}, got {matches.size}"
        )
    return int(matches[0])


def phish_probability_from_proba_row(
    proba_row: Union[np.ndarray, Sequence[float]],
    classes: Union[np.ndarray, Sequence],
) -> float:
    """P(phishing) aligned with :data:`INTERNAL_PHISH` and the fitted ``classes_`` order."""
    idx = class_index_for_internal_label(classes, INTERNAL_PHISH)
    return float(np.asarray(proba_row, dtype=float)[idx])


def kaggle_status_to_internal(status: int) -> int:
    """Map Kaggle-style 1=legit, 0=phish to internal 0=legit, 1=phish."""
    if int(status) == 1:
        return INTERNAL_LEGIT
    if int(status) == 0:
        return INTERNAL_PHISH
    raise ValueError(f"Expected binary status 0 or 1, got {status!r}")
