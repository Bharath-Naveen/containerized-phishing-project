"""One place that decides what a URL "is" for training, grouping and inference.

Why this module exists (rebuild, Phase 1):
  The Kaggle dump mixes full URLs ("https://x.com/a") with bare hosts ("x.com/a"). Earlier code
  sometimes fed bare hosts straight into feature extraction and grouping, so the parser saw no
  hostname at all. Training features then differed from what the app computes at inference, and
  the "domain-grouped" split put one URL per group instead of one domain per group.

Every caller now goes through these two functions:
  canonical_url(raw)  -> the stored, deduplicated form (always has a scheme)
  feature_url(raw)    -> the scheme-neutral form features are computed from

Scheme-neutral: in the Kaggle data an explicit https:// is far more common on phishing rows than
on legitimate ones, while on the real web almost everything is https. Letting the model see the
scheme teaches it a dataset artifact, so features are computed with the scheme fixed to http://.
"""

from __future__ import annotations

from typing import Tuple

from src.pipeline.safe_url import canonicalize_url_safe

FEATURE_SCHEME = "http://"


def canonical_url(raw: str) -> Tuple[str, int, str]:
    """(canonical_url, invalid_flag, parse_error). Adds http:// when the scheme is missing."""
    return canonicalize_url_safe(raw)


def original_scheme(raw: str) -> str:
    """Scheme the URL actually had ('' when it had none)."""
    s = (raw or "").strip()
    if "://" not in s:
        return ""
    return s.split("://", 1)[0].strip().lower()


def feature_url(raw: str) -> str:
    """Canonical URL with the scheme replaced by http:// so no feature can depend on it."""
    canon, _inv, _err = canonical_url(raw)
    if not canon:
        return ""
    rest = canon.split("://", 1)[1] if "://" in canon else canon
    return FEATURE_SCHEME + rest
