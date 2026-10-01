"""Shared host/domain helpers and constants used by the capture signals, verdict rules and EAL.

Split out of the former analyze_dashboard.py in rebuild Phase 2 (code moved unchanged).
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import tldextract
from phishguard.features.brand_signals import BRAND_TOKENS



# Fallback only: if raw model score is still high on a host under a known official registrable family.
_OFFICIAL_BRAND_APEX_PHISH_CAP = 0.32


_FREE_HOSTING_SUFFIXES = (
    "vercel.app",
    "netlify.app",
    "github.io",
    "pages.dev",
    "workers.dev",
    "firebaseapp.com",
    "web.app",
    "cloudfront.net",
    "azurewebsites.net",
    "herokuapp.com",
)


_BUILDER_PLATFORM_SUFFIXES = (
    "framer.app",
    "webflow.io",
    "notion.site",
    "wixsite.com",
    "squarespace.com",
    "typedream.app",
    "carrd.co",
)


_CREATOR_PLATFORM_SUFFIXES = (
    "podia.com",
    "gumroad.com",
    "shopify.com",
    "myshopify.com",
    "notion.site",
    "webflow.io",
    "squarespace.com",
    "wixsite.com",
)


_URL_WEAK_KEYWORDS = ("login", "verify", "secure", "account", "update", "payment", "auth", "wallet")


_SUSPICIOUS_HTML_KEYWORDS = ("verify", "password", "account", "security", "urgent", "suspend", "billing")


_IMPERSONATION_BRAND_TOKENS = set(BRAND_TOKENS) | {"handshake"}


def _is_major_brand_token(token: str) -> bool:
    t = (token or "").strip().lower()
    return bool(t and t in _IMPERSONATION_BRAND_TOKENS)


def _contains_major_brand_token(text: str) -> bool:
    blob = f" {str(text or '').lower()} "
    for tok in _IMPERSONATION_BRAND_TOKENS:
        t = str(tok or "").strip().lower()
        if len(t) < 3:
            continue
        if f" {t} " in blob or f"-{t}" in blob or f"{t}-" in blob or f"/{t}" in blob or f"{t}/" in blob or f".{t}" in blob:
            return True
    return False


def _normalize_entity_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _creator_platform_host_candidate(
    *,
    final_host: str,
    final_reg: str,
    title: str,
    visible_text: str,
) -> Tuple[bool, str, List[str]]:
    reasons: List[str] = []
    if not final_host or not final_reg or final_reg not in _CREATOR_PLATFORM_SUFFIXES:
        return False, "", reasons
    if final_host == final_reg or not final_host.endswith("." + final_reg):
        return False, "", reasons
    sub = final_host[: -(len(final_reg) + 1)].strip().lower()
    primary = sub.split(".")[0].strip()
    if len(primary) < 3:
        return False, "", reasons
    if _is_major_brand_token(primary):
        reasons.append("subdomain_matches_major_brand_token")
        return False, primary, reasons
    primary_norm = _normalize_entity_token(primary)
    text_blob = f"{(title or '').lower()} {(visible_text or '').lower()}"
    text_norm = _normalize_entity_token(text_blob)
    if primary_norm and primary_norm in text_norm:
        reasons.append("subdomain_brand_matches_page_content")
        return True, primary, reasons
    # Ecosystem consistency: allow creator brand to match linked/mentioned same-entity apex domains.
    domain_mentions = set(re.findall(r"\b[a-z0-9-]+\.[a-z]{2,}\b", text_blob))
    for dm in domain_mentions:
        reg = _reg_domain(dm)
        if not reg:
            continue
        reg_label = _normalize_entity_token(reg.split(".")[0])
        if primary_norm and reg_label and (primary_norm in reg_label or reg_label in primary_norm):
            reasons.append("subdomain_brand_matches_ecosystem_domain")
            return True, primary, reasons
    reasons.append("subdomain_brand_not_reflected_in_page_content")
    return False, primary, reasons


def _reg_domain(url_or_host: str) -> str:
    raw = (url_or_host or "").strip()
    try:
        host = (urlparse(raw).hostname or "").lower() if "://" in raw else raw.lower()
    except Exception:
        host = raw.lower()
    ext = tldextract.extract(host)
    return ".".join(p for p in (ext.domain, ext.suffix) if p).lower()


def _hostname(url_or_host: str) -> str:
    raw = (url_or_host or "").strip()
    try:
        if "://" in raw:
            return (urlparse(raw).hostname or "").lower()
    except Exception:
        return ""
    return raw.lower()


def _detect_cloud_hosted_brand_impersonation(
    *,
    final_registered_domain: str,
    final_host: str,
    final_url: str,
) -> bool:
    reg = (final_registered_domain or "").lower()
    if reg not in _FREE_HOSTING_SUFFIXES:
        return False
    host_l = (final_host or "").lower()
    path_l = (urlparse(final_url).path or "").lower() if final_url else ""
    for tok in _IMPERSONATION_BRAND_TOKENS:
        t = str(tok or "").strip().lower()
        if len(t) < 3:
            continue
        if t in host_l or t in path_l:
            return True
    return False


def _safe_subdomain_allowed(host: str, registered_domain: str) -> bool:
    h = (host or "").lower().strip(".")
    rd = (registered_domain or "").lower().strip(".")
    if not h or not rd:
        return False
    return h == rd or h.endswith("." + rd)


def _is_strong_brand_domain_mismatch(
    layer2_capture: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]] = None,
) -> bool:
    cap = layer2_capture or {}
    dom = html_dom_summary or {}
    if bool(dom.get("trust_surface_brand_domain_mismatch")):
        return True
    strength = str(cap.get("brand_domain_mismatch_strength") or "").lower()
    return bool(
        strength == "strong"
        and bool(cap.get("brand_domain_mismatch"))
    )
