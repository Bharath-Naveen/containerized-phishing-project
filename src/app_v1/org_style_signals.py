"""Organization-style reinforcement heuristics (Layer 2): hosting, brand/domain, forms, path, text.

Used as **secondary** evidence on top of the Layer-1 ML model — not a standalone detector.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .legit_lookup import TRUSTED_BRAND_ROOT_FALLBACK, is_url_on_trusted_brand_root

_FREE_HOSTING = (
    ".vercel.app",
    ".github.io",
    ".gitlab.io",
    ".netlify.app",
    ".pages.dev",
    ".web.app",
    ".firebaseapp.com",
    ".herokuapp.com",
    ".azurewebsites.net",
    ".blogspot.com",
    ".wixsite.com",
    ".weebly.com",
    ".cloudfront.net",
)

_AUTH_PATH = re.compile(
    r"(login|signin|sign-in|verify|verification|oauth|sso|password|reset|account|billing|checkout|wallet)",
    re.I,
)

_SENSITIVE_TEXT = (
    "sign in",
    "log in",
    "verify your account",
    "unusual activity",
    "confirm your identity",
    "suspended",
    "update your payment",
    "enter password",
    "security check",
)


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def free_hosting_hit(url: str) -> Optional[str]:
    u = (url or "").lower()
    for suf in _FREE_HOSTING:
        if suf in u:
            return suf
    return None


def brand_claim_without_trusted_domain(final_url: str, visible_text: str, title: str) -> List[str]:
    """If page/title mentions a major brand but hostname is not on that brand's trusted roots."""
    blob = f"{title}\n{visible_text}".lower()
    hits: List[str] = []
    for brand in TRUSTED_BRAND_ROOT_FALLBACK.keys():
        if brand in blob and not is_url_on_trusted_brand_root(brand, final_url):
            hits.append(brand)
    return hits


def org_style_score(
    *,
    input_url: str,
    final_url: str,
    title: str = "",
    visible_text: str = "",
    password_field_found: bool = False,
    form_count_hint: int = 0,
    off_domain_favicon_flag: int = 0,
    capture_error: Optional[str] = None,
    page_fetch_ok: bool = True,
) -> Dict[str, Any]:
    """Return structured signals + coarse risk score 0..1."""
    reasons: List[str] = []
    score = 0.0

    fh = free_hosting_hit(final_url or input_url)
    if fh:
        score += 0.25
        reasons.append(f"Host matches free/static hosting pattern ({fh.strip('.')}).")

    path = urlparse(final_url or input_url).path or ""
    if _AUTH_PATH.search(path + " " + (input_url or "")):
        score += 0.05
        reasons.append("URL path suggests auth / verification / payment flow.")

    brand_miss = brand_claim_without_trusted_domain(final_url or input_url, visible_text, title)
    for b in brand_miss:
        score += 0.2
        reasons.append(f"Content references “{b}” but hostname is not on known {b} first-party domains.")

    if password_field_found:
        score += 0.1
        reasons.append("Password field detected in DOM (login-style surface).")

    if form_count_hint and form_count_hint >= 2:
        score += 0.05
        reasons.append("Multiple forms present (higher complexity / possible clone).")

    if int(off_domain_favicon_flag or 0) == 1:
        score += 0.1
        reasons.append("Favicon/hosting hints suggest off-domain branding.")

    low = (visible_text or "").lower()
    for phrase in _SENSITIVE_TEXT:
        if phrase in low:
            score += 0.05
            reasons.append(f"Sensitive phrase in visible text: “{phrase}”.")
            break

    if capture_error or not page_fetch_ok:
        reasons.append("Live page content was limited or unavailable; org-style checks are partial.")

    score = min(1.0, score)
    triggered = bool(
        fh or brand_miss or password_field_found or int(off_domain_favicon_flag or 0) == 1
    )
    return {
        "org_style_risk_score": round(score, 4),
        "org_style_triggered": triggered,
        "reasons": reasons,
        "free_hosting_match": fh,
        "brand_domain_mismatches": brand_miss,
    }


def org_style_from_capture_blob(capture: Dict[str, Any], input_url: str) -> Dict[str, Any]:
    """Consume JSON ``capture`` dict (as from :meth:`CaptureResult.as_json`)."""
    final_url = str(capture.get("final_url") or input_url)
    title = str(capture.get("title") or "")
    vis = str(capture.get("visible_text") or "")
    pw = bool((capture.get("interaction") or {}).get("password_field_found"))
    err = capture.get("error")
    blocked = bool(capture.get("capture_blocked"))
    strat = str(capture.get("capture_strategy") or "")
    fetch_ok = not err and not blocked and strat != "failed"
    return org_style_score(
        input_url=input_url,
        final_url=final_url,
        title=title,
        visible_text=vis,
        password_field_found=pw,
        form_count_hint=0,
        off_domain_favicon_flag=0,
        capture_error=str(err) if err else None,
        page_fetch_ok=fetch_ok,
    )
