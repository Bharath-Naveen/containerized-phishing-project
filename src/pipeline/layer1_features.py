"""Fast Layer-1 features: URL lexical + hosting (optional DNS). No HTTP fetch / no browser."""

from __future__ import annotations

from typing import Any, Dict, List, Set
from urllib.parse import urlsplit

from src.pipeline.features.dns_features import extract_dns_features
from src.pipeline.features.hosting_features import extract_hosting_features
from src.pipeline.features.url_features import extract_url_features


def _host_from_url(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def layer1_feature_key_set(*, include_dns: bool) -> Set[str]:
    keys = set(extract_layer1_features("https://sub.example.com/path?q=1", use_dns=include_dns).keys())
    keys.discard("canonical_url")
    return keys


def extract_layer1_features(canonical_url: str, *, use_dns: bool = False) -> Dict[str, Any]:
    url = (canonical_url or "").strip()
    row: Dict[str, Any] = {"canonical_url": url}
    row.update(extract_url_features(url))
    row.update(extract_hosting_features(url))
    # Avoid ultra-high-cardinality string for sklearn one-hot; keep numeric bucket.
    reg = str(row.pop("registered_domain", "") or "")
    row["domain_hash_bucket"] = abs(hash(reg)) % 2048
    if use_dns:
        row["dns_features_skipped"] = 0
        row.update(extract_dns_features(_host_from_url(url)))
    else:
        row["dns_features_skipped"] = 1
    row["url_features_missing"] = 0
    row["hosting_features_missing"] = int(not _host_from_url(url))
    return row


def layer1_numeric_allowlist(*, include_dns: bool) -> List[str]:
    """Ordered feature keys for training (excluding canonical_url)."""
    sample = extract_layer1_features("https://example.com/login", use_dns=include_dns)
    return [k for k in sorted(sample.keys()) if k != "canonical_url"]
