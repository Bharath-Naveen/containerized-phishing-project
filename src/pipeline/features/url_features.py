"""Lexical URL features."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict
from urllib.parse import urlparse

_BRANDS = ("amazon", "google", "microsoft", "paypal", "apple", "facebook", "netflix")
_KEYWORD_FLAGS = (
    ("kw_login", ("login", "signin", "sign-in", "log-in")),
    ("kw_verify", ("verify", "verification", "validate")),
    ("kw_secure", ("secure", "ssl", "safe")),
    ("kw_update", ("update", "upgrade")),
    ("kw_account", ("account", "profile", "billing")),
    ("kw_payment", ("payment", "checkout", "pay", "invoice")),
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


def extract_url_features(url: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    raw = (url or "").strip()
    out["url_length"] = len(raw)
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if "@" in host:
        host = host.split("@")[-1]
    path = parsed.path or ""
    query = parsed.query or ""

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
    out["has_https"] = int(parsed.scheme.lower() == "https")
    parts = host.split(".")
    out["tld"] = ".".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else "")

    out["hostname_entropy"] = round(_shannon_entropy(host), 4)

    blob = f"{host} {path} {query}".lower()
    out["brand_token_in_domain"] = int(any(b in host for b in _BRANDS))
    out["brand_token_in_path"] = int(any(b in path.lower() for b in _BRANDS))

    suspicious_tokens = (
        "verify", "secure", "update", "suspend", "locked", "unusual", "confirm",
        "wallet", "invoice", "tax", "refund", "prize", "winner",
    )
    out["suspicious_keyword_count"] = sum(blob.count(t) for t in suspicious_tokens)
    for col, words in _KEYWORD_FLAGS:
        out[col] = int(any(w in blob for w in words))
    return out
