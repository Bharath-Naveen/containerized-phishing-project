"""Compare suspicious page capture against a trusted legitimate reference capture.

Lightweight, deterministic signals only (no heavy CV). Visual similarity remains a TODO.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .schemas import CaptureInteractionMetadata, CaptureResult, ComparisonResult

_VISIBLE_TEXT_COMPARE_MAX = 6000

_LOGINISH_TASKS = frozenset({"login", "password reset", "account verification"})

# Known-good federated identity / OAuth provider hosts (normalized, no leading www.).
TRUSTED_AUTH_DOMAINS_LIST: List[str] = [
    "accounts.google.com",
    "appleid.apple.com",
    "login.microsoftonline.com",
]
TRUSTED_AUTH_DOMAINS = frozenset(TRUSTED_AUTH_DOMAINS_LIST)


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _read_html_text(html_path: str) -> str:
    return Path(html_path).read_text(encoding="utf-8", errors="ignore")


def _safe_domain(url: str) -> str:
    d = _domain(url)
    return re.sub(r"^www\.", "", d)


def _surface_family(url: str) -> str:
    u = (url or "").lower()
    d = _safe_domain(url)
    if "lens.google" in d or "visualsearch" in u or "#homework" in u:
        return "google_lens"
    if "accounts.google.com" in d:
        return "google_accounts"
    if "google." in d:
        return "google_web"
    if "outlook." in d:
        return "microsoft_outlook"
    if "onedrive." in d:
        return "microsoft_onedrive"
    if "sharepoint." in d:
        return "microsoft_sharepoint"
    if "login.microsoftonline.com" in d:
        return "microsoft_auth"
    if "paypal." in d:
        return "paypal_web"
    if "amazon." in d:
        return "amazon_web"
    if "facebook.com" in d:
        return "facebook_web"
    if "stripe.com" in d:
        return "stripe_web"
    return d or "unknown"


def _reference_match_tier(suspicious_url: str, reference_url: str) -> str:
    sus_d = _safe_domain(suspicious_url)
    ref_d = _safe_domain(reference_url)
    if not ref_d:
        return "unknown"
    if sus_d and ref_d and sus_d == ref_d:
        if _surface_family(suspicious_url) == _surface_family(reference_url):
            return "exact_surface"
        return "same_product"
    sus_f = _surface_family(suspicious_url)
    ref_f = _surface_family(reference_url)
    if sus_f != "unknown" and ref_f != "unknown":
        if sus_f.split("_")[0] == ref_f.split("_")[0]:
            return "same_product"
    if sus_d and ref_d and (sus_d.split(".")[-2:] == ref_d.split(".")[-2:]):
        return "same_brand_fallback"
    return "weak_fallback"


def _host_is_trusted_auth(after_host: str) -> bool:
    h = _safe_domain(f"https://{after_host}/") if after_host and "://" not in after_host else _safe_domain(after_host)
    if not h:
        return False
    if h in TRUSTED_AUTH_DOMAINS:
        return True
    for d in TRUSTED_AUTH_DOMAINS:
        if h == d or h.endswith("." + d):
            return True
    return False


def _oauth_url_signals(url: str) -> bool:
    """Heuristic: URL looks like part of an OAuth / SSO redirect chain."""
    u = (url or "").lower()
    if "accounts.google.com" in u:
        return True
    if "oauth" in u:
        return True
    if "redirect_uri" in u:
        return True
    return False


def _analyze_post_submit_redirect(
    interaction: CaptureInteractionMetadata,
    trusted_primary_url: Optional[str],
) -> tuple[Optional[bool], bool, bool]:
    """Return (suspicious_cross_domain_redirect, is_oauth_flow, trusted_oauth_redirect)."""
    if not interaction.attempted_submit or not interaction.url_after_submit:
        return None, False, False
    after_url = interaction.url_after_submit
    after_host = _safe_domain(after_url)
    is_oauth = _oauth_url_signals(after_url)
    trusted_auth = _host_is_trusted_auth(after_host)

    if not trusted_primary_url:
        # No reference host to leave; still record OAuth signals for consumers.
        return None, is_oauth, trusted_auth

    trusted_host = _safe_domain(trusted_primary_url)
    if not trusted_host or not after_host:
        return None, is_oauth, trusted_auth

    if after_host == trusted_host:
        return False, is_oauth, trusted_auth

    # Cross-domain from trusted reference landing.
    if trusted_auth or is_oauth:
        # Legitimate federated login often leaves the service domain for Google / Apple / Microsoft, etc.
        return False, is_oauth, trusted_auth

    return True, is_oauth, trusted_auth


def _similarity_ratio(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(float(SequenceMatcher(None, a, b).ratio()), 4)


def _similarity_ratio_with_missing_reason(
    left: str,
    right: str,
    *,
    metric_name: str,
    reasons: List[str],
) -> float:
    l = (left or "").strip()
    r = (right or "").strip()
    if not l and not r:
        reasons.append(
            f"{metric_name}_comparison_unavailable_missing_both_sides"
        )
        return 0.0
    if not l or not r:
        if not l and r:
            reasons.append(f"{metric_name}_comparison_unavailable_missing_suspicious_side")
        elif l and not r:
            reasons.append(f"{metric_name}_comparison_unavailable_missing_legit_side")
        reasons.append(
            f"{metric_name}_comparison_unavailable_missing_one_side"
        )
        return 0.0
    return _similarity_ratio(l, r)


def _login_interaction_possible(meta: CaptureInteractionMetadata) -> bool:
    return bool(meta.user_field_found and meta.password_field_found)


def _dom_similarity_score(suspicious_html_path: str, legit_html_path: str) -> tuple[float, list[str]]:
    """Compute a lightweight DOM structure similarity score (0..1)."""
    reasons: list[str] = []
    if not suspicious_html_path or not Path(suspicious_html_path).is_file():
        reasons.append("dom_similarity_skipped_missing_suspicious_html")
        return 0.0, reasons
    if not legit_html_path or not Path(legit_html_path).is_file():
        reasons.append("dom_similarity_skipped_missing_legit_html")
        return 0.0, reasons
    suspicious_html = _read_html_text(suspicious_html_path)
    legit_html = _read_html_text(legit_html_path)

    soup_a = BeautifulSoup(suspicious_html, "html.parser")
    soup_b = BeautifulSoup(legit_html, "html.parser")

    for tag in soup_a(["script", "style", "noscript", "template"]):
        tag.decompose()
    for tag in soup_b(["script", "style", "noscript", "template"]):
        tag.decompose()

    tags_a = [t.name for t in soup_a.find_all(True)]
    tags_b = [t.name for t in soup_b.find_all(True)]

    set_a = set(tags_a)
    set_b = set(tags_b)
    union = set_a | set_b
    intersection = set_a & set_b
    jaccard = (len(intersection) / len(union)) if union else 0.0
    if union:
        reasons.append(f"dom_jaccard={jaccard:.3f}")

    forms_a = len(soup_a.find_all("form"))
    forms_b = len(soup_b.find_all("form"))
    form_match = 1.0 if forms_a == forms_b else 0.0

    links_a = len(soup_a.find_all("a"))
    links_b = len(soup_b.find_all("a"))
    denom = max(links_a, links_b, 1)
    link_diff = abs(links_a - links_b) / denom
    link_score = 1.0 - min(link_diff, 1.0)

    similarity = 0.6 * jaccard + 0.2 * form_match + 0.2 * link_score
    similarity = float(max(0.0, min(1.0, similarity)))
    return similarity, reasons


def _task_structurally_aligned(
    task_guess: str,
    trusted_reference_found: bool,
    sus_pw: bool,
    legit_pw: bool,
) -> bool:
    if not trusted_reference_found:
        return False
    t = (task_guess or "").strip().lower()
    if t in _LOGINISH_TASKS:
        return sus_pw and legit_pw
    # checkout, document share, informational, unknown — no strong password heuristic
    return True


def _behavior_gap_score(
    title_sim: float,
    text_sim: float,
    sus_pw: bool,
    legit_pw: bool,
    sus_login: bool,
    legit_login: bool,
    post_left: Optional[bool],
) -> float:
    gap = 0.0
    gap += 0.22 * (1.0 - title_sim)
    gap += 0.22 * (1.0 - text_sim)
    if sus_pw != legit_pw:
        gap += 0.18
    if sus_login != legit_login:
        gap += 0.18
    if post_left is True:
        gap += 0.2
    return float(round(min(1.0, gap), 4))


def _plain_english_reasons(
    *,
    title_similarity: float,
    title_available: bool,
    visible_text_similarity: float,
    visible_text_available: bool,
    trusted_reference_found: bool,
    suspicious_password_field_present: bool,
    legit_password_field_present: bool,
    suspicious_login_interaction_possible: bool,
    legit_login_interaction_possible: bool,
    post_submit_left_trusted_domain: Optional[bool],
    is_oauth_flow: bool,
    trusted_oauth_redirect: bool,
    task_aligned_with_legit_reference: bool,
    reasons: List[str],
) -> List[str]:
    out: List[str] = []

    if not title_available:
        if reasons and "title_comparison_unavailable_missing_suspicious_side" in reasons:
            out.append(
                "Title comparison unavailable because suspicious capture was missing."
            )
        else:
            out.append(
                "Title comparison unavailable because both captures lacked page titles."
            )
    else:
        if title_similarity < 0.25:
            out.append(
                "The suspicious page title looks very different from the legitimate reference page title."
            )
        elif title_similarity < 0.5:
            out.append("The page titles only partially match the trusted reference.")

    if not visible_text_available:
        if reasons and "visible_text_comparison_unavailable_missing_suspicious_side" in reasons:
            out.append(
                "Visible-text comparison unavailable because suspicious capture was missing."
            )
        else:
            out.append(
                "Visible-text comparison unavailable because both captures lacked extracted text."
            )
    else:
        if visible_text_similarity < 0.2:
            out.append(
                "The visible text on the suspicious page is quite different from the legitimate reference."
            )
        elif visible_text_similarity < 0.45:
            out.append(
                "The visible text shows noticeable differences compared to the legitimate reference."
            )

    if trusted_reference_found:
        if suspicious_password_field_present and not legit_password_field_present:
            out.append(
                "The suspicious page shows a password field, but the trusted reference capture does not."
            )
        elif legit_password_field_present and not suspicious_password_field_present:
            out.append(
                "The trusted reference shows a password field, but the suspicious page capture does not."
            )

        if suspicious_login_interaction_possible and not legit_login_interaction_possible:
            out.append(
                "User and password fields appear together on the suspicious page, but not on the trusted reference capture."
            )
        elif legit_login_interaction_possible and not suspicious_login_interaction_possible:
            out.append(
                "The trusted reference exposes both user and password fields, but the suspicious capture does not."
            )

    if trusted_oauth_redirect:
        out.append(
            "Redirected to trusted OAuth provider (e.g., Google, Apple, or Microsoft) — expected behavior."
        )
    elif is_oauth_flow and post_submit_left_trusted_domain is not True:
        out.append(
            "Post-submit URL shows OAuth-style patterns (oauth, redirect_uri, or Google accounts); "
            "cross-domain navigation treated as lower risk."
        )

    if post_submit_left_trusted_domain is True:
        out.append(
            "After a dummy login submit on the suspicious page, the URL moved to a host "
            "that does not match the trusted reference domain and is not a recognized OAuth or identity provider."
        )

    if trusted_reference_found and not task_aligned_with_legit_reference:
        out.append(
            "Structural signals suggest the suspicious page may not be the same kind of task "
            "as the trusted reference for this brand."
        )

    return out


def compare_suspicious_vs_legit_reference(
    *,
    suspicious: CaptureResult,
    legit: Optional[CaptureResult],
    brand_guess: str,
    task_guess: str,
    trusted_reference_found: bool,
    matched_legit_urls: list[str],
    url_product_hint: str = "unknown",
    url_action_hint: str = "unknown",
) -> ComparisonResult:
    """Compare suspicious capture vs legit reference capture using structured fields."""
    reasons: list[str] = []
    try:
        if suspicious.capture_blocked:
            reasons.append(
                "Target site blocked automated browsing attempts, which is commonly associated "
                "with phishing or evasive infrastructure."
            )
        if not suspicious.settled_successfully:
            reasons.append(
                "Suspicious screenshot was captured from best-effort rendered state because networkidle settle did not fully complete."
            )

        suspicious_final_url = suspicious.final_url
        legit_final_url = legit.final_url if legit else None
        suspicious_html_path = suspicious.html_path
        legit_html_path = legit.html_path if legit else None

        suspicious_domain = _safe_domain(suspicious_final_url or "")
        legit_domain = _safe_domain(legit_final_url or "") if legit_final_url else ""

        brand_domain_match = bool(
            trusted_reference_found
            and suspicious_domain
            and legit_domain
            and suspicious_domain == legit_domain
        )
        task_alignment = bool(
            trusted_reference_found
            and (task_guess or "").strip().lower() not in {"", "unknown"}
        )

        action_match_score = 0.0
        if trusted_reference_found:
            action_match_score = round(
                (0.6 * (1.0 if brand_domain_match else 0.0))
                + (0.4 * (1.0 if task_alignment else 0.0)),
                3,
            )

        sus_i = suspicious.interaction
        legit_i = legit.interaction if legit else CaptureInteractionMetadata()

        suspicious_password_field_present = bool(sus_i.password_field_found)
        legit_password_field_present = bool(legit_i.password_field_found)
        suspicious_login_interaction_possible = _login_interaction_possible(sus_i)
        legit_login_interaction_possible = _login_interaction_possible(legit_i)

        trusted_primary = matched_legit_urls[0] if matched_legit_urls else None
        legit_reference_match_tier = _reference_match_tier(
            suspicious_final_url, trusted_primary or ""
        )
        (
            post_submit_left_trusted_domain,
            is_oauth_flow,
            trusted_oauth_redirect,
        ) = _analyze_post_submit_redirect(sus_i, trusted_primary)
        url_product_action_aligned: Optional[bool] = None
        if url_product_hint not in {"", "unknown"} or url_action_hint not in {"", "unknown"}:
            ref_surface = _surface_family(trusted_primary or "")
            product_ok = True
            action_ok = True
            if url_product_hint not in {"", "unknown"}:
                product_ok = url_product_hint.split("_")[0] == ref_surface.split("_")[0] or url_product_hint in ref_surface
            if url_action_hint not in {"", "unknown"}:
                action_ok = (task_guess or "").strip().lower() in {url_action_hint, "unknown", ""}
            url_product_action_aligned = bool(product_ok and action_ok)
            if not url_product_action_aligned:
                reasons.append(
                    "The selected legit reference does not appear to match the product/action implied by the input URL."
                )
                reasons.append(
                    "Comparison may not be apples-to-apples because URL hints suggest a different surface."
                )

        left_title = suspicious.title
        right_title = legit.title if legit else ""
        title_similarity = _similarity_ratio_with_missing_reason(
            left_title,
            right_title,
            metric_name="title",
            reasons=reasons,
        )
        sus_txt = (suspicious.visible_text or "")[:_VISIBLE_TEXT_COMPARE_MAX]
        leg_txt = ((legit.visible_text if legit else "") or "")[:_VISIBLE_TEXT_COMPARE_MAX]
        visible_text_similarity = _similarity_ratio_with_missing_reason(
            sus_txt,
            leg_txt,
            metric_name="visible_text",
            reasons=reasons,
        )
        title_available = bool((left_title or "").strip() and (right_title or "").strip())
        visible_text_available = bool((sus_txt or "").strip() and (leg_txt or "").strip())

        task_aligned_with_legit_reference = _task_structurally_aligned(
            task_guess,
            trusted_reference_found,
            suspicious_password_field_present,
            legit_password_field_present,
        )
        sus_lang = (suspicious.detected_language or "").strip().lower()
        leg_lang = ((legit.detected_language if legit else None) or "").strip().lower()
        language_match: Optional[bool] = None
        if sus_lang and leg_lang:
            language_match = sus_lang == leg_lang
            if not language_match:
                reasons.append(
                    f"language_mismatch: suspicious={sus_lang} legit={leg_lang} (weak signal only)"
                )
        legit_reference_quality = "unknown"
        legit_reference_matches_intended_task: Optional[bool] = None
        if legit:
            lt = f"{legit.title} {legit.visible_text}".lower()
            if any(t in lt for t in ("something went wrong", "service unavailable", "access denied", "temporarily unavailable")):
                legit_reference_quality = "error_page"
            elif any(t in lt for t in ("cookie", "consent", "interstitial")):
                legit_reference_quality = "interstitial"
            elif not lt.strip():
                legit_reference_quality = "unknown"
            else:
                legit_reference_quality = "valid"
            task = (task_guess or "").strip().lower()
            if task in {"login", "password reset", "account verification"}:
                legit_reference_matches_intended_task = any(
                    t in lt for t in ("sign in", "login", "password", "verify", "reset")
                )
            elif task == "checkout":
                legit_reference_matches_intended_task = any(t in lt for t in ("checkout", "payment", "pay"))
            elif task:
                legit_reference_matches_intended_task = True
            if legit_reference_quality in {"error_page", "interstitial"}:
                reasons.append(
                    f"legit_reference_quality={legit_reference_quality}; comparison confidence reduced"
                )

        dom_similarity_score = 0.0
        if trusted_reference_found and legit_html_path:
            dom_similarity_score, dom_reasons = _dom_similarity_score(
                suspicious_html_path=suspicious_html_path,
                legit_html_path=legit_html_path,
            )
            reasons.extend(dom_reasons)
            if "dom_similarity_skipped_missing_suspicious_html" in dom_reasons:
                reasons.append(
                    "HTML structure comparison unavailable because suspicious HTML was missing."
                )
        elif not trusted_reference_found:
            reasons.append("dom_similarity_skipped_no_trusted_reference")
        else:
            reasons.append("dom_similarity_skipped_missing_reference_html")

        visual_similarity_score = 0.0
        reasons.append(
            "Image-based visual similarity is not available for this URL, so the verdict used deterministic signals only."
        )

        behavior_gap_score = _behavior_gap_score(
            title_similarity,
            visible_text_similarity,
            suspicious_password_field_present,
            legit_password_field_present,
            suspicious_login_interaction_possible,
            legit_login_interaction_possible,
            post_submit_left_trusted_domain,
        )

        plain = _plain_english_reasons(
            title_similarity=title_similarity,
            title_available=title_available,
            visible_text_similarity=visible_text_similarity,
            visible_text_available=visible_text_available,
            trusted_reference_found=trusted_reference_found,
            suspicious_password_field_present=suspicious_password_field_present,
            legit_password_field_present=legit_password_field_present,
            suspicious_login_interaction_possible=suspicious_login_interaction_possible,
            legit_login_interaction_possible=legit_login_interaction_possible,
            post_submit_left_trusted_domain=post_submit_left_trusted_domain,
            is_oauth_flow=is_oauth_flow,
            trusted_oauth_redirect=trusted_oauth_redirect,
            task_aligned_with_legit_reference=task_aligned_with_legit_reference,
            reasons=reasons,
        )
        reasons.extend(plain)

        if trusted_reference_found:
            reasons.append(f"brand_domain_match={brand_domain_match}")
            reasons.append(f"task_alignment={task_alignment}")
            if legit_reference_match_tier not in {"exact_surface", "same_product"}:
                reasons.append(
                    "The selected legitimate reference is from the same broader brand family but may not be the exact product surface."
                )
                reasons.append(
                    "Comparison confidence is reduced because the reference is not the same product/page type (comparison may not be apples-to-apples due to different product surface)."
                )
                behavior_gap_score = float(round(max(0.0, behavior_gap_score * 0.8), 4))
        else:
            reasons.append("No trusted reference URL was found for this brand and task guess.")

        return ComparisonResult(
            brand_guess=brand_guess,
            task_guess=task_guess,
            trusted_reference_found=bool(trusted_reference_found),
            matched_legit_urls=matched_legit_urls,
            title_similarity=title_similarity,
            visible_text_similarity=visible_text_similarity,
            suspicious_password_field_present=suspicious_password_field_present,
            legit_password_field_present=legit_password_field_present,
            suspicious_login_interaction_possible=suspicious_login_interaction_possible,
            legit_login_interaction_possible=legit_login_interaction_possible,
            post_submit_left_trusted_domain=post_submit_left_trusted_domain,
            is_oauth_flow=is_oauth_flow,
            trusted_oauth_redirect=trusted_oauth_redirect,
            task_aligned_with_legit_reference=task_aligned_with_legit_reference,
            language_match=language_match,
            legit_reference_matches_intended_task=legit_reference_matches_intended_task,
            legit_reference_quality=legit_reference_quality,
            legit_reference_match_tier=legit_reference_match_tier,
            url_product_action_aligned=url_product_action_aligned,
            action_match_score=float(action_match_score),
            visual_similarity_score=float(visual_similarity_score),
            dom_similarity_score=float(dom_similarity_score),
            behavior_gap_score=float(behavior_gap_score),
            reasons=reasons,
        )
    except Exception as exc:  # noqa: BLE001
        reasons.append(
            "Comparison stage hit an unexpected error; deterministic fallback values were used."
        )
        reasons.append(
            "Image-based visual similarity is not available for this URL, so the verdict used deterministic signals only."
        )
        return ComparisonResult(
            brand_guess=brand_guess,
            task_guess=task_guess,
            trusted_reference_found=bool(trusted_reference_found),
            matched_legit_urls=matched_legit_urls,
            reasons=reasons,
            error=f"comparison_exception: {type(exc).__name__}: {exc}",
        )
