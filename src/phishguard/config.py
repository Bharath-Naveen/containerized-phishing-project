"""Project-wide defaults in one place (rebuild Phase 2).

Paths come from :mod:`phishguard.paths` (overridable with PHISH_* environment variables).
Runtime capture settings live in :mod:`phishguard.app.runtime_config`.
"""

from __future__ import annotations

SEED = 42                       # every sample, split, model and bootstrap uses this
DEFAULT_SAMPLE_SIZE = 50_000    # default stratified Kaggle sample (use --full for all ~796K rows)
TEST_FRACTION = 0.2             # domain-grouped test split
VALIDATION_FRACTION = 0.15      # domain-grouped validation split inside the training rows
PRIMARY_SELECTION_POLICY = "validated"  # best validation PR-AUC (see models/train.py)
