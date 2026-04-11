"""Extract lightweight heuristic features from captured page artifacts."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .legit_lookup import TRUSTED_BRAND_ROOT_FALLBACK, is_url_on_trusted_brand_root
from .schemas import FeatureResult


def _safe_domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _url_auth_token_signal(url: str) -> bool:
    u = (url or "").lower()
    tokens = (
        "login",
        "signin",
        "sign-in",
        "verify",
        "secure",
        "account",
        "auth",
        "sso",
        "oauth",
        "password",
        "mfa",
        "2fa",
    )
    return any(tok in u for tok in tokens)


def _extract_host_tokens(host: str) -> list[str]:
    parts = re.split(r"[.\-_]+", (host or "").lower())
    return [p for p in parts if p and len(p) >= 4]


def _brand_lookalike_signal(host: str) -> tuple[bool, Optional[str]]:
    host_tokens = _extract_host_tokens(host)
    if not host_tokens:
        return False, None
    for brand in TRUSTED_BRAND_ROOT_FALLBACK.keys():
        b = brand.lower()
        for token in host_tokens:
            if token == b:
                continue
            ratio = SequenceMatcher(None, token, b).ratio()
            # Close but not exact token (e.g., gooogle, micr0soft) suggests lookalike host labeling.
            if ratio >= 0.8 and token[0] == b[0]:
                return True, b
    return False, None


def _trusted_brand_root_mismatch(final_url: str) -> bool:
    host = _safe_domain(final_url)
    if not host:
        return False
    low = final_url.lower()
    for brand in TRUSTED_BRAND_ROOT_FALLBACK.keys():
        if brand in host or brand in low:
            if not is_url_on_trusted_brand_root(brand, final_url):
                return True
    return False


def extract_features(
    input_url: str,
    final_url: str,
    title: str,
    visible_text: str,
    html_path: str,
    *,
    capture_blocked: bool = False,
    capture_block_reason: Optional[str] = None,
    capture_block_evidence: Optional[str] = None,
    screenshot_path: str = "",
    redirect_count: int = 0,
    redirect_chain: Optional[list[str]] = None,
    cross_domain_redirect_count: int = 0,
    settled_successfully: bool = False,
    settle_time_ms: int = 0,
    suspicious_language: Optional[str] = None,
    legit_language: Optional[str] = None,
    task_guess: str = "",
    legit_title: str = "",
    legit_visible_text: str = "",
    legit_reference_match_tier: str = "unknown",
    url_brand_hint: str = "unknown",
    url_product_hint: str = "unknown",
    url_action_hint: str = "unknown",
    url_language_hint: Optional[str] = None,
    url_first_party_plausibility: str = "unknown",
    url_shape_reasons: Optional[list[str]] = None,
) -> FeatureResult:
    """Compute heuristic features with robust fallbacks."""
    final_domain = _safe_domain(final_url)
    has_form = False
    external_ratio = 0.0
    visual_capture_unavailable = not bool(
        str(screenshot_path or "").strip()
        and str(html_path or "").strip()
        and str(visible_text or "").strip()
    )

    try:
        html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        has_form = soup.find("form") is not None
        links = [a.get("href", "") for a in soup.find_all("a")]
        links = [href for href in links if href]

        if links:
            external = 0
            for href in links:
                target_domain = _safe_domain(href)
                if target_domain and target_domain != final_domain:
                    external += 1
                elif href.startswith("http://") or href.startswith("https://"):
                    # Absolute URL without parsable domain should still count suspiciously.
                    external += 1
            external_ratio = round(external / len(links), 4)
    except Exception:
        # Keep defaults and continue; orchestration should not fail hard.
        pass

    clean_text = re.sub(r"\s+", " ", visible_text or "").strip()
    auth_tokens = _url_auth_token_signal(final_url or input_url)
    lookalike_signal, lookalike_to = _brand_lookalike_signal(final_domain)
    trusted_root_mismatch = _trusted_brand_root_mismatch(final_url or input_url)
    language_match: Optional[bool]
    if suspicious_language and legit_language:
        language_match = suspicious_language.lower() == legit_language.lower()
    else:
        language_match = None
    task = (task_guess or "").strip().lower()
    legit_text_blob = f"{legit_title} {legit_visible_text}".lower()
    degraded_tokens = (
        "something went wrong",
        "service unavailable",
        "access denied",
        "temporarily unavailable",
        "cookie",
        "consent",
        "interstitial",
        "sorry, we couldn't find that page",
        "dogs of amazon",
        "storage is full",
        "account suspended",
        "billing issue",
        "suspicious activity",
    )
    if any(t in legit_text_blob for t in degraded_tokens):
        legit_reference_quality = "error_page" if any(
            t in legit_text_blob for t in ("went wrong", "unavailable", "access denied")
        ) else "interstitial"
    elif task in {
        "login",
        "checkout",
        "account verification",
        "password reset",
        "informational",
    } and not any(
        t in legit_text_blob
        for t in (
            "sign in",
            "login",
            "password",
            "checkout",
            "verify",
            "reset",
            "order",
            "shipment",
            "delivery",
            "billing",
            "storage",
            "suspension",
        )
    ):
        legit_reference_quality = "partial"
    elif not legit_text_blob.strip():
        legit_reference_quality = "unknown"
    else:
        legit_reference_quality = "valid"
    legit_reference_matches_intended_task: Optional[bool] = None
    if task:
        if legit_reference_quality in {"error_page", "interstitial"}:
            legit_reference_matches_intended_task = False
        elif task in {"login", "password reset", "account verification"}:
            legit_reference_matches_intended_task = any(
                t in legit_text_blob for t in ("sign in", "login", "password", "verify", "reset")
            )
        elif task == "checkout":
            legit_reference_matches_intended_task = any(
                t in legit_text_blob for t in ("checkout", "payment", "pay")
            )
        elif task == "informational":
            legit_reference_matches_intended_task = any(
                t in legit_text_blob
                for t in (
                    "order",
                    "shipment",
                    "delivery",
                    "billing",
                    "storage",
                    "suspended",
                    "verification",
                )
            )
        else:
            legit_reference_matches_intended_task = True
    return FeatureResult(
        input_url=input_url,
        final_url=final_url,
        final_domain=final_domain,
        has_form=has_form,
        external_link_ratio=external_ratio,
        title_length=len(title or ""),
        visible_text_length=len(clean_text),
        capture_blocked=capture_blocked,
        capture_block_reason=capture_block_reason,
        capture_block_evidence=capture_block_evidence,
        visual_capture_unavailable=visual_capture_unavailable,
        url_contains_auth_tokens=auth_tokens,
        brand_lookalike_signal=lookalike_signal,
        brand_lookalike_to=lookalike_to,
        trusted_brand_root_mismatch=trusted_root_mismatch,
        redirect_count=max(0, int(redirect_count or 0)),
        redirect_chain=list(redirect_chain or []),
        cross_domain_redirect_count=max(0, int(cross_domain_redirect_count or 0)),
        settled_successfully=bool(settled_successfully),
        settle_time_ms=max(0, int(settle_time_ms or 0)),
        suspicious_language=suspicious_language,
        legit_language=legit_language,
        language_match=language_match,
        legit_reference_matches_intended_task=legit_reference_matches_intended_task,
        legit_reference_quality=legit_reference_quality,
        legit_reference_match_tier=legit_reference_match_tier or "unknown",
        url_brand_hint=url_brand_hint or "unknown",
        url_product_hint=url_product_hint or "unknown",
        url_action_hint=url_action_hint or "unknown",
        url_language_hint=url_language_hint,
        url_first_party_plausibility=url_first_party_plausibility or "unknown",
        url_shape_reasons=list(url_shape_reasons or []),
    )
