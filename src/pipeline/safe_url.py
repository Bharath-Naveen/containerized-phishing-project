"""Defensive URL parsing — never raise on malformed strings (e.g. invalid IPv6 in netloc).

Used by clean, lexical/hosting features, Layer-1, and leak-safe grouping.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Tuple
from urllib.parse import ParseResult, SplitResult, parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

# urlparse / urlsplit can raise ValueError: Invalid IPv6 URL (and similar) on hostile inputs.


def stable_url_fingerprint(url: str) -> str:
    """Short deterministic id for fallback group keys."""
    b = (url or "").encode("utf-8", errors="replace")
    return hashlib.blake2s(b, digest_size=8).hexdigest()


def safe_urlsplit(url: str) -> Tuple[Optional[SplitResult], Optional[str]]:
    try:
        return urlsplit(url or ""), None
    except ValueError as e:
        return None, str(e)


def safe_urlparse(url: str) -> Tuple[Optional[ParseResult], Optional[str]]:
    try:
        return urlparse(url or ""), None
    except ValueError as e:
        return None, str(e)


def safe_hostname(url: str) -> Tuple[str, Optional[str]]:
    """
    Hostname suitable for tldextract / grouping.
    Returns ("", error_message) when URL cannot be parsed safely.
    """
    parts, err = safe_urlsplit(url)
    if err or parts is None:
        return "", err
    try:
        hn = parts.hostname
        return ((hn or "").lower(), None)
    except ValueError as e:
        return "", str(e)


def canonicalize_url_safe(raw: str) -> Tuple[str, int, str]:
    """
    Same contract as :func:`src.pipeline.clean.canonicalize_url` but routes parsing through
    :func:`safe_urlsplit` so malformed URLs never raise.
    """
    s = (raw or "").strip()
    if not s:
        return "", 1, "empty"
    s = re.sub(r"[\s\u200b\u00a0]+", "", s)
    if "://" not in s:
        s = "http://" + s
    parts, perr = safe_urlsplit(s)
    if perr or parts is None:
        return s, 1, f"split_error:{perr}"

    scheme = (parts.scheme or "http").lower()
    netloc = (parts.netloc or "").lower()
    if not netloc:
        return s, 1, "missing_host"
    path = parts.path or ""
    if path.endswith("/") and path != "/":
        path = path.rstrip("/")
    query = parts.query or ""
    fragment = parts.fragment or ""

    if query:
        try:
            q_pairs = parse_qsl(query, keep_blank_values=True)
            query = urlencode(q_pairs)
        except Exception:
            pass

    canon = urlunsplit((scheme, netloc, path, query, fragment))
    invalid = 0
    err = ""
    if scheme not in {"http", "https"}:
        invalid = 1
        err = "unsupported_scheme"
    if ".." in path:
        invalid = 1
        err = "suspicious_path"
    return canon, invalid, err


def leak_safe_group_key(canonical_url: str) -> Tuple[str, int]:
    """
    Registered-domain style key for StratifiedGroupKFold, never raises.

    Returns:
        (group_key, group_key_fallback_used) where fallback is 1 if we used ``malformed::<hash>``.
    """
    import tldextract

    url = (canonical_url or "").strip()
    host, perr = safe_hostname(url)
    if perr or not host:
        return f"malformed::{stable_url_fingerprint(url)}", 1
    try:
        ext = tldextract.extract(host)
        reg = ".".join(p for p in (ext.domain, ext.suffix) if p)
        key = (reg.lower() or host).strip()
        if not key:
            return f"malformed::{stable_url_fingerprint(url)}", 1
        return key, 0
    except Exception:
        return f"malformed::{stable_url_fingerprint(url)}", 1


def netloc_path_query_from_url(url: str) -> Tuple[str, str, str, str, Optional[str]]:
    """
    Lexical feature helper: (netloc_lower, path, query, scheme_lower, parse_error).
    On failure returns ("", "", "", "", err).
    """
    p, err = safe_urlparse(url or "")
    if err or p is None:
        return "", "", "", "", err
    netloc = (p.netloc or "").lower()
    if "@" in netloc:
        netloc = netloc.split("@")[-1]
    scheme = (p.scheme or "").lower()
    return netloc, p.path or "", p.query or "", scheme, None
