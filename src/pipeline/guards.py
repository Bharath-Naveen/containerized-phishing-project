"""Hard checks that stop the pipeline instead of letting it produce misleading numbers.

Added in the rebuild (Phase 1) after the audit found two silent failures in the shipped models:
  * split leak: scheme-less URLs got a per-URL ``malformed::`` group key, so the same domain
    landed in both train and test (55% of "held-out" rows shared a domain with training);
  * train/serve skew: stored training features differed from what the app computes.

Each guard raises ``PipelineGuardError`` with a plain-language message.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd


class PipelineGuardError(RuntimeError):
    pass


MAX_MALFORMED_GROUP_FRACTION = 0.005  # at most 0.5% of rows may lack a real registered domain
MIN_MALFORMED_ROWS_TO_FAIL = 10  # tiny fixtures with 1-2 junk URLs should not trip the guard


def check_group_keys(keys: Iterable[str], *, where: str, max_fraction: float = MAX_MALFORMED_GROUP_FRACTION) -> Dict[str, float]:
    keys = list(keys)
    n = len(keys)
    bad = sum(1 for k in keys if str(k).startswith("malformed::"))
    frac = bad / n if n else 0.0
    if frac > max_fraction and bad >= MIN_MALFORMED_ROWS_TO_FAIL:
        raise PipelineGuardError(
            f"[{where}] {bad} of {n} rows ({frac:.1%}) have no parsable registered domain. "
            f"Limit is {max_fraction:.1%}. Grouping by domain would silently degrade to one group per URL."
        )
    return {"rows": n, "malformed_group_keys": bad, "malformed_fraction": frac}


def check_no_group_overlap(train_keys: Iterable[str], test_keys: Iterable[str], *, where: str) -> int:
    shared: Set[str] = set(train_keys) & set(test_keys)
    if shared:
        sample = ", ".join(sorted(shared)[:5])
        raise PipelineGuardError(f"[{where}] {len(shared)} registered domains appear in both train and test (e.g. {sample}).")
    return 0


def check_feature_parity(
    df: pd.DataFrame,
    feature_columns: List[str],
    *,
    n: int = 500,
    seed: int = 42,
    where: str = "train",
) -> Dict[str, int]:
    """Recompute features for a sample of stored rows exactly as the app does and compare."""
    from src.pipeline.layer1_features import extract_layer1_features

    if df.empty or "canonical_url" not in df.columns:
        return {"checked_rows": 0, "mismatched_rows": 0}
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    mismatched: List[str] = []
    cols = [c for c in feature_columns if c in sample.columns]
    for _, row in sample.iterrows():
        fresh = extract_layer1_features(str(row["canonical_url"]))
        for c in cols:
            a, b = row[c], fresh.get(c)
            if _differs(a, b):
                mismatched.append(f"{row['canonical_url']} :: {c} stored={a!r} app={b!r}")
                break
    if mismatched:
        raise PipelineGuardError(
            f"[{where}] train/serve feature skew in {len(mismatched)} of {len(sample)} sampled rows. First: {mismatched[0]}"
        )
    return {"checked_rows": int(len(sample)), "mismatched_rows": 0}


def _differs(a, b) -> bool:
    def num(x) -> Optional[float]:
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)) or str(x) in ("", "nan", "None"):
                return None
            return float(x)
        except (TypeError, ValueError):
            return None

    na, nb = num(a), num(b)
    if na is not None or nb is not None:
        if na is None or nb is None:
            return True
        return abs(na - nb) > 1e-6
    return str(a) != str(b)
