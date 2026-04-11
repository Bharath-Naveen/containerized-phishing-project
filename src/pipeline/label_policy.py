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

INTERNAL_LEGIT = 0
INTERNAL_PHISH = 1


def kaggle_status_to_internal(status: int) -> int:
    """Map Kaggle-style 1=legit, 0=phish to internal 0=legit, 1=phish."""
    if int(status) == 1:
        return INTERNAL_LEGIT
    if int(status) == 0:
        return INTERNAL_PHISH
    raise ValueError(f"Expected binary status 0 or 1, got {status!r}")
