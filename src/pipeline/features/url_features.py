"""Lexical URL features (non-brand; brand structure lives in :mod:`brand_signals`)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict
from urllib.parse import parse_qsl

from src.pipeline.features.brand_signals import host_on_official_brand_apex
from src.pipeline.safe_url import netloc_path_query_from_url

# Re-export for callers that imported from url_features.
__all__ = ["extract_url_features", "host_on_official_brand_apex"]

_KEYWORD_FLAGS = (
    ("kw_login", ("login", "signin", "sign-in", "log-in")),
    ("kw_verify", ("verify", "verification", "validate")),
    ("kw_secure", ("secure", "ssl", "safe")),
    ("kw_update", ("update", "upgrade")),
    ("kw_account", ("account", "profile", "billing")),
    # Avoid matching "pay" inside legitimate hostnames like ``paypal.com``.
    ("kw_payment", ("payment", "checkout", "invoice")),
    ("kw_confirm", ("confirm", "confirmation")),
    ("kw_password", ("password", "passwd", "reset-password")),
)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _count_special(s: str) -> int:
    return sum(1 for ch in s if not ch.isalnum() and ch not in ".-")


def _path_segment_count(path: str) -> int:
    """Non-empty path segments (``/a/b`` → 2; ``/`` or empty → 0)."""
    p = (path or "").strip("/")
    if not p:
        return 0
    return len([x for x in p.split("/") if x])


def _suspicious_redirect_query_flag(query: str) -> int:
    """Open-redirect / hop-style query keys common in phishing chains."""
    if not query:
        return 0
    try:
        keys = {k.lower().strip() for k, _ in parse_qsl(query, keep_blank_values=False)}
    except Exception:
        return 0
    risky = {
        "url",
        "redirect",
        "redirect_uri",
        "return",
        "return_url",
        "next",
        "continue",
        "dest",
        "destination",
        "goto",
        "rurl",
        "target",
        "redir",
        "link",
        "out",
        "forward",
    }
    return int(bool(keys & risky))


def extract_url_features(url: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    raw = (url or "").strip()
    out["url_length"] = len(raw)
    host, path, query, scheme, _perr = netloc_path_query_from_url(raw)

    out["hostname_length"] = len(host)
    out["path_length"] = len(path)
    out["query_length"] = len(query)
    out["num_dots"] = host.count(".")
    out["num_hyphens"] = raw.count("-")
    out["num_digits"] = sum(1 for ch in raw if ch.isdigit())
    out["num_special_chars"] = _count_special(host)
    labels = host.split(".") if host else []
    out["subdomain_count"] = max(0, len(labels) - 2) if len(labels) >= 2 else 0
    out["has_ip_address"] = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host.split(":")[0])))
    out["has_at_symbol"] = int("@" in raw)
    out["has_https"] = int(scheme == "https")
    parts = host.split(".")
    out["tld"] = ".".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "")

    out["hostname_entropy"] = round(_shannon_entropy(host), 4)

    _nseg = _path_segment_count(path)
    out["path_segment_count"] = _nseg
    out["path_shallow_le1"] = int(_nseg <= 1)
    out["path_shallow_le2"] = int(_nseg <= 2)
    out["suspicious_redirect_query_flag"] = _suspicious_redirect_query_flag(query)

    blob = f"{host} {path} {query}".lower()
    path_query_blob = f"{path} {query}".lower()
    suspicious_tokens = (
        "verify", "secure", "update", "suspend", "locked", "unusual", "confirm",
        "wallet", "invoice", "tax", "refund", "prize", "winner",
    )
    out["suspicious_keyword_count"] = sum(blob.count(t) for t in suspicious_tokens)
    # Login/payment cues: path+query only so ``login.microsoftonline.com`` / ``paypal.com`` are not false positives.
    for col, words in _KEYWORD_FLAGS:
        out[col] = int(any(w in path_query_blob for w in words))
    _authish_hits = sum(int(out.get(col, 0) or 0) for col, _ in _KEYWORD_FLAGS)
    out["path_query_authish_keyword_hits"] = int(_authish_hits)
    out["no_authish_path_query_tokens"] = int(_authish_hits == 0)
    return out
