"""Look up expected legitimate URLs for (brand, task) pairs."""

from __future__ import annotations

from typing import Dict, List, Tuple
from urllib.parse import urlparse

from .schemas import LegitLookupResult


# Hardcoded trusted mapping for triage enrichment (exact brand + task).
TRUSTED_BRAND_TASK_URLS: Dict[Tuple[str, str], List[str]] = {
    ("microsoft", "login"): ["https://login.microsoftonline.com/"],
    ("microsoft", "homepage"): ["https://www.microsoft.com/"],
    ("microsoft", "password reset"): ["https://passwordreset.microsoftonline.com/"],
    ("microsoft", "account verification"): ["https://account.microsoft.com/"],
    ("google", "login"): ["https://accounts.google.com/"],
    ("google", "homepage"): ["https://www.google.com/"],
    ("google", "password reset"): ["https://accounts.google.com/signin/v2/usernamerecovery"],
    ("google", "account verification"): ["https://myaccount.google.com/security"],
    ("google_lens", "homepage"): ["https://lens.google/"],
    ("google_gmail", "homepage"): ["https://mail.google.com/"],
    ("google_drive", "homepage"): ["https://drive.google.com/"],
    ("google_photos", "homepage"): ["https://photos.google.com/"],
    ("apple", "login"): ["https://appleid.apple.com/"],
    ("apple", "homepage"): ["https://www.apple.com/"],
    ("apple", "password reset"): ["https://iforgot.apple.com/"],
    ("facebook", "login"): ["https://www.facebook.com/login/"],
    ("facebook", "account verification"): ["https://www.facebook.com/checkpoint/"],
    ("paypal", "checkout"): ["https://www.paypal.com/checkoutnow"],
    ("paypal", "login"): ["https://www.paypal.com/signin"],
    ("paypal", "account verification"): ["https://www.paypal.com/myaccount/security/"],
    ("amazon", "checkout"): ["https://www.amazon.com/checkout/"],
    ("amazon", "login"): ["https://www.amazon.com/ap/signin"],
    ("amazon", "homepage"): ["https://www.amazon.com/"],
    ("amazon", "browse"): ["https://www.amazon.com/"],
    ("amazon", "account verification"): ["https://www.amazon.com/ap/cnep"],
    ("amazon", "password reset"): ["https://www.amazon.com/ap/forgotpassword"],
    ("amazon", "informational"): ["https://www.amazon.com/your-orders"],
    ("stripe", "checkout"): ["https://checkout.stripe.com/"],
    ("dropbox", "document share"): ["https://www.dropbox.com/"],
    ("docusign", "document share"): ["https://www.docusign.com/"],
    ("github", "login"): ["https://github.com/login"],
    ("github", "homepage"): ["https://github.com/"],
}

# When (brand, task) is missing, fall back to official brand roots for domain checks and capture.
TRUSTED_BRAND_ROOT_FALLBACK: Dict[str, List[str]] = {
    "amazon": ["https://www.amazon.com/"],
    "microsoft": [
        "https://www.microsoft.com/",
        "https://login.microsoftonline.com/",
    ],
    "paypal": ["https://www.paypal.com/"],
    "google": ["https://www.google.com/", "https://accounts.google.com/"],
    "apple": ["https://www.apple.com/", "https://appleid.apple.com/"],
    "facebook": ["https://www.facebook.com/"],
    "stripe": ["https://stripe.com/", "https://checkout.stripe.com/"],
    "github": ["https://github.com/"],
    "dropbox": ["https://www.dropbox.com/"],
    "docusign": ["https://www.docusign.com/"],
}


def _host_key(url: str) -> str:
    try:
        h = (urlparse(url).netloc or "").lower()
    except Exception:
        return ""
    if h.startswith("www."):
        return h[4:]
    return h


def _surface_from_url(url: str) -> str:
    u = (url or "").lower()
    host = _host_key(u)
    if "lens.google" in host or "/#homework" in u or "visualsearch" in u or "/lens" in u:
        return "google_lens"
    if "accounts.google.com" in host:
        return "google_accounts"
    if "mail.google.com" in host or "gmail" in u:
        return "google_gmail"
    if "drive.google.com" in host or "drive" in u:
        return "google_drive"
    if "photos.google.com" in host or "photos" in u:
        return "google_photos"
    if "google." in host:
        return "google_web"
    if "outlook.live.com" in host or "outlook.office.com" in host:
        return "microsoft_outlook"
    if "onedrive.live.com" in host or "onedrive" in u:
        return "microsoft_onedrive"
    if "sharepoint.com" in host or "sharepoint" in u:
        return "microsoft_sharepoint"
    if "amazon." in host:
        return "amazon_web"
    if "paypal." in host:
        return "paypal_web"
    if "facebook.com" in host:
        return "facebook_web"
    if "checkout.stripe.com" in host or "stripe" in host:
        return "stripe_checkout"
    return "unknown"


def _reference_match_tier(input_url: str, reference_url: str, brand_guess: str) -> str:
    in_host = _host_key(input_url)
    ref_host = _host_key(reference_url)
    if in_host and ref_host and input_url.strip() and reference_url.strip():
        if in_host == ref_host:
            in_surface = _surface_from_url(input_url)
            ref_surface = _surface_from_url(reference_url)
            if in_surface != "unknown" and in_surface == ref_surface:
                return "exact_surface"
            return "same_product"
    brand = (brand_guess or "").strip().lower()
    if brand and ref_host and (brand in ref_host or ref_host.endswith(f".{brand}.com")):
        return "same_brand_fallback"
    if ref_host:
        return "weak_fallback"
    return "unknown"


def _append_product_candidates(urls: List[str], brand: str, surface: str, task: str) -> None:
    # Prefer same brand -> same product/surface -> same task.
    if brand == "google":
        if surface in {"google_lens", "google_gmail", "google_drive", "google_photos"}:
            urls.extend(TRUSTED_BRAND_TASK_URLS.get((surface, "homepage"), []))
        if task in {"login", "password reset", "account verification"}:
            urls.extend(TRUSTED_BRAND_TASK_URLS.get(("google", task), []))
    if brand == "microsoft":
        if surface == "microsoft_outlook":
            urls.append("https://outlook.office.com/")
        if surface == "microsoft_onedrive":
            urls.append("https://onedrive.live.com/")
        if surface == "microsoft_sharepoint":
            urls.append("https://www.microsoft.com/microsoft-365/sharepoint/collaboration")
        if task in {"login", "password reset", "account verification"}:
            urls.extend(TRUSTED_BRAND_TASK_URLS.get(("microsoft", task), []))
    if brand == "paypal" and task in {"login", "checkout", "account verification"}:
        urls.extend(TRUSTED_BRAND_TASK_URLS.get(("paypal", task), []))
    if brand == "amazon":
        if "order" in surface or "order" in task or task == "informational":
            urls.append("https://www.amazon.com/your-orders")
        if task in {"homepage", "login", "checkout", "account verification", "password reset", "informational"}:
            urls.extend(TRUSTED_BRAND_TASK_URLS.get(("amazon", task), []))
    if brand == "facebook" and task in {"login", "account verification"}:
        urls.extend(TRUSTED_BRAND_TASK_URLS.get(("facebook", task), []))
    if brand == "stripe" and task == "checkout":
        urls.extend(TRUSTED_BRAND_TASK_URLS.get(("stripe", "checkout"), []))


def is_url_on_trusted_brand_root(brand_guess: str, final_url: str) -> bool:
    """True if final_url's host matches a known-good root for this brand (exact or subdomain)."""
    brand = (brand_guess or "").strip().lower()
    if brand == "unknown" or not brand:
        return False
    host = _host_key(final_url or "")
    if not host:
        return False

    roots: List[str] = []
    roots.extend(TRUSTED_BRAND_ROOT_FALLBACK.get(brand, []))
    for (b, _t), urls in TRUSTED_BRAND_TASK_URLS.items():
        if b == brand:
            roots.extend(urls)

    seen: set[str] = set()
    for u in roots:
        if u in seen:
            continue
        seen.add(u)
        rh = _host_key(u)
        if not rh:
            continue
        if host == rh or host.endswith("." + rh):
            return True
    return False


def lookup_legitimate_urls(
    brand_guess: str,
    task_guess: str,
    *,
    input_url: str = "",
    product_hint: str = "unknown",
    action_hint: str = "unknown",
    language_hint: str | None = None,
) -> LegitLookupResult:
    """Return candidate legitimate URLs: exact (brand, task) first, else brand root fallback."""
    try:
        brand = (brand_guess or "unknown").strip().lower()
        task = (task_guess or "unknown").strip().lower()
        # Product/surface-aware overrides first.
        in_surface = _surface_from_url(input_url)
        inferred_brand = brand
        surface_to_brand = {
            "google_lens": "google",
            "google_accounts": "google",
            "google_gmail": "google",
            "google_drive": "google",
            "google_photos": "google",
            "google_web": "google",
            "microsoft_outlook": "microsoft",
            "microsoft_onedrive": "microsoft",
            "microsoft_sharepoint": "microsoft",
            "amazon_web": "amazon",
            "paypal_web": "paypal",
            "facebook_web": "facebook",
            "stripe_checkout": "stripe",
        }
        if inferred_brand in {"", "unknown"}:
            if product_hint not in {"", "unknown"}:
                inferred_brand = surface_to_brand.get(product_hint, inferred_brand)
            if inferred_brand in {"", "unknown"}:
                inferred_brand = surface_to_brand.get(in_surface, inferred_brand)
        effective_task = task if task not in {"unknown", ""} else (action_hint or "unknown")
        urls: List[str] = []
        if in_surface == "google_lens" and inferred_brand in {"google", "unknown", ""}:
            # Prefer Lens surface when suspicious URL indicates lens/homework/visual search.
            urls.extend(TRUSTED_BRAND_TASK_URLS.get(("google_lens", "homepage"), []))
            if effective_task in {"homepage", "unknown", ""}:
                urls.extend(TRUSTED_BRAND_TASK_URLS.get(("google", "homepage"), []))
        elif inferred_brand in {"google", "microsoft", "paypal", "amazon", "facebook", "stripe"}:
            _append_product_candidates(
                urls,
                inferred_brand,
                product_hint if product_hint not in {"", "unknown"} else in_surface,
                effective_task,
            )

        if not urls:
            urls = list(TRUSTED_BRAND_TASK_URLS.get((inferred_brand, effective_task), []))
        if not urls:
            urls = list(TRUSTED_BRAND_ROOT_FALLBACK.get(inferred_brand, []))
        # Deduplicate while preserving order.
        deduped: List[str] = []
        seen: set[str] = set()
        for u in urls:
            if u not in seen:
                seen.add(u)
                deduped.append(u)
        selected = deduped[0] if deduped else None
        tier = _reference_match_tier(input_url, selected or "", inferred_brand)
        return LegitLookupResult(
            candidate_urls=deduped,
            matched=bool(deduped),
            selected_reference_url=selected,
            legit_reference_match_tier=tier,
        )
    except Exception as exc:  # noqa: BLE001 - Defensive behavior.
        return LegitLookupResult(error=f"legit_lookup_failed: {exc}")
