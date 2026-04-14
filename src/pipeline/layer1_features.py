"""Fast Layer-1 features: URL lexical + hosting (optional DNS). No HTTP fetch / no browser."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Set

from src.pipeline.features.brand_signals import extract_brand_structure_features
from src.pipeline.features.dns_features import extract_dns_features
from src.pipeline.features.hosting_features import extract_hosting_features
from src.pipeline.features.url_features import extract_url_features
from src.pipeline.safe_url import netloc_path_query_from_url, safe_hostname


def _host_from_url(url: str) -> str:
    h, _ = safe_hostname(url)
    return h


def _stable_domain_bucket(registered_domain: str) -> int:
    """Deterministic bucket (reproducible across runs; avoids Python's salted ``hash()``)."""
    if not registered_domain:
        return 0
    digest = hashlib.blake2s(registered_domain.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 2048


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
    _host, path, _, _, _ = netloc_path_query_from_url(url)
    row.update(
        extract_brand_structure_features(
            host=_host,
            path=path,
            registered_domain=reg,
            free_hosting_flag=int(row.get("free_hosting_flag", 0) or 0),
            cloud_hosting_flag=int(row.get("cloud_hosting_flag", 0) or 0),
        )
    )
    row["domain_hash_bucket"] = _stable_domain_bucket(reg)
    # Learnable trust signal: official registrable + clean brand label (down-weight if hyphen-deception).
    _o = int(row.get("official_registrable_anchor", 0) or 0)
    _e = int(row.get("brand_hostname_exact_label_match", 0) or 0)
    _h = int(row.get("brand_hyphenated_deception_label", 0) or 0)
    if _h:
        row["layer1_brand_trust_score"] = _o * 2
    else:
        row["layer1_brand_trust_score"] = _o * (2 + _e)
    _https = int(row.get("has_https", 0) or 0)
    row["official_anchor_with_https"] = int(_https * _o)
    row["https_without_official_anchor"] = int(_https * (1 - _o))
    pl = (path or "").lower()
    _auth_any = any(int(row.get(k, 0) or 0) for k in ("kw_login", "kw_verify", "kw_account", "kw_password"))
    row["legit_auth_surface_on_official_anchor"] = int(_o and _auth_any)
    _adm = any(
        x in pl
        for x in (
            "/admin",
            "/dashboard",
            "/console",
            "/portal",
            "/wp-admin",
            "/settings",
            "/security",
        )
    )
    row["legit_admin_like_path_on_official_anchor"] = int(_o and _adm)
    row["legit_checkout_like_path_on_official_anchor"] = int(_o and int(row.get("kw_payment", 0) or 0))
    _nseg = int(row.get("path_segment_count", 99) or 0)
    _shallow = int(_nseg <= 1)
    _no_authish = int(row.get("no_authish_path_query_tokens", 0) or 0)
    _no_redir_q = int(1 - int(row.get("suspicious_redirect_query_flag", 0) or 0))
    _not_ip = int(1 - int(row.get("has_ip_address", 0) or 0))
    row["simple_public_web_shape"] = int(_shallow and _no_authish and _no_redir_q and _not_ip)
    row["simple_official_homepage_shape"] = int(_o and _shallow and _no_authish and _no_redir_q and _not_ip)
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
