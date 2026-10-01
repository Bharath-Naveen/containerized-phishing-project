"""Turns a live capture (Playwright result + HTML) into evidence: security block pages, OAuth buttons, language, capture-failure classification.

Split out of the former analyze_dashboard.py in rebuild Phase 2 (code moved unchanged).
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from phishguard.features.brand_signals import host_on_official_brand_apex
from .utils.download_models import resolve_fasttext_model_path
from .domain_utils import (
    _CREATOR_PLATFORM_SUFFIXES,
    _FREE_HOSTING_SUFFIXES,
    _SUSPICIOUS_HTML_KEYWORDS,
    _URL_WEAK_KEYWORDS,
    _contains_major_brand_token,
    _creator_platform_host_candidate,
    _hostname,
    _reg_domain,
)
from .brand_coherence import (
    compute_brand_domain_coherence,
)


_FASTTEXT_MIN_TEXT_CHARS = 120


_FASTTEXT_MODEL_CACHE: Any = None


_FASTTEXT_MODEL_ERROR: Optional[str] = None


_SECURITY_BLOCK_VENDOR_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "lionic": (
        "block.cloud.lionic.com",
        "/blockpage/malicious.html",
        "warning: visiting this site may harm your device",
        "this page has been blocked",
        "might steal your confidential information",
        "security consideration",
    ),
}


def _load_fasttext_language_model() -> Any:
    global _FASTTEXT_MODEL_CACHE
    global _FASTTEXT_MODEL_ERROR
    if _FASTTEXT_MODEL_CACHE is not None or _FASTTEXT_MODEL_ERROR is not None:
        return _FASTTEXT_MODEL_CACHE
    model_path = resolve_fasttext_model_path()
    try:
        import fasttext  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _FASTTEXT_MODEL_ERROR = f"fasttext_import_error:{type(exc).__name__}"
        return None
    p = Path(model_path)
    if not p.is_file():
        _FASTTEXT_MODEL_ERROR = "fasttext_model_not_found"
        return None
    try:
        _FASTTEXT_MODEL_CACHE = fasttext.load_model(str(p))
    except Exception as exc:  # noqa: BLE001
        _FASTTEXT_MODEL_ERROR = f"fasttext_model_load_error:{type(exc).__name__}"
        return None
    return _FASTTEXT_MODEL_CACHE


def _detect_security_block_page(capture_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cj = capture_json or {}
    final_url = str(cj.get("final_url") or "").lower()
    title = str(cj.get("title") or "").lower()
    visible = str(cj.get("visible_text") or cj.get("visible_text_sample") or "").lower()
    blob = "\n".join(x for x in (final_url, title, visible) if x)
    for vendor, pats in _SECURITY_BLOCK_VENDOR_PATTERNS.items():
        matched = [p for p in pats if p in blob]
        if matched:
            return {
                "security_block_page_detected": True,
                "security_block_page_vendor": vendor,
                "security_block_page_reasons": matched,
            }
    return {
        "security_block_page_detected": False,
        "security_block_page_vendor": None,
        "security_block_page_reasons": [],
    }


def _fasttext_language_enrichment(capture_json: Optional[Dict[str, Any]], soup: Any) -> Dict[str, Any]:
    cj = capture_json or {}
    source_text = str(cj.get("visible_text") or "").strip()
    if not source_text and soup is not None:
        try:
            source_text = " ".join((soup.get_text(" ", strip=True) or "").split())
        except Exception:
            source_text = ""
    out: Dict[str, Any] = {
        "detected_language": None,
        "detected_language_confidence": None,
        "language_detection_available": False,
        "language_detection_error": None,
        "language_mismatch_contextual_signal": False,
    }
    if not source_text:
        return out
    if len(source_text) < _FASTTEXT_MIN_TEXT_CHARS:
        out["language_detection_error"] = "insufficient_text"
        return out
    model = _load_fasttext_language_model()
    if model is None:
        out["language_detection_error"] = _FASTTEXT_MODEL_ERROR
        return out
    try:
        labels, probs = model.predict(source_text.replace("\n", " "), k=1)
        label = str(labels[0]) if labels else ""
        lang = label.replace("__label__", "").strip().lower() if label else ""
        conf = float(probs[0]) if probs else None
        out["detected_language"] = lang or None
        out["detected_language_confidence"] = round(conf, 4) if isinstance(conf, float) else None
        out["language_detection_available"] = bool(lang)
    except Exception as exc:  # noqa: BLE001
        out["language_detection_error"] = f"language_detect_failed:{type(exc).__name__}"
        return out
    baseline = str(cj.get("detected_language") or "").strip().lower()
    detected = str(out.get("detected_language") or "").strip().lower()
    if baseline and detected and baseline != detected:
        out["language_mismatch_contextual_signal"] = True
    return out


def _detect_oauth_providers(capture_json: Optional[Dict[str, Any]], soup: Any) -> Dict[str, Any]:
    cj = capture_json or {}
    oauth_patterns: Dict[str, Tuple[str, ...]] = {
        "google": (
            r"\blog[\s-]*in with google\b",
            r"\bsign[\s-]*in with google\b",
            r"\bcontinue with google\b",
            r"\bcontinue using google\b",
            r"\bgoogle\b.{0,24}\b(sign[\s-]*in|continue|oauth|sso)\b",
        ),
        "facebook": (
            r"\blog[\s-]*in with facebook\b",
            r"\bsign[\s-]*in with facebook\b",
            r"\bcontinue with facebook\b",
            r"\bcontinue using facebook\b",
            r"\bfacebook\b.{0,24}\b(sign[\s-]*in|continue|oauth|sso)\b",
        ),
        "apple": (
            r"\blog[\s-]*in with apple\b",
            r"\bsign[\s-]*in with apple\b",
            r"\bcontinue with apple\b",
            r"\bcontinue using apple\b",
            r"\bapple\b.{0,24}\b(sign[\s-]*in|continue|oauth|sso)\b",
        ),
        "github": (
            r"\blog[\s-]*in with github\b",
            r"\bsign[\s-]*in with github\b",
            r"\bcontinue with github\b",
            r"\bcontinue using github\b",
            r"\bgithub\b.{0,24}\b(sign[\s-]*in|continue|oauth|sso)\b",
        ),
        "square": (
            r"\blog[\s-]*in with square\b",
            r"\bsign[\s-]*in with square\b",
            r"\bcontinue with square\b",
            r"\bcontinue using square\b",
            r"\bsquare\b.{0,24}\b(sign[\s-]*in|continue|oauth|sso)\b",
        ),
    }
    oauth_link_domains: Dict[str, Tuple[str, ...]] = {
        "google": ("accounts.google.com",),
        "facebook": ("facebook.com",),
        "apple": ("appleid.apple.com",),
        "github": ("github.com",),
        "square": ("squareup.com", "square.com"),
    }
    text_chunks: List[str] = []
    links_blob: List[str] = []
    text_chunks.append(str(cj.get("visible_text") or ""))
    text_chunks.append(str(cj.get("title") or ""))
    if soup is not None:
        try:
            text_chunks.append(str(soup.get_text(" ", strip=True) or ""))
            for tag in soup.find_all(["button", "a", "input", "div", "span"]):
                bits: List[str] = []
                bits.append(str(tag.get_text(" ", strip=True) or ""))
                for attr in ("aria-label", "alt", "title", "value", "name", "id", "class"):
                    val = tag.get(attr)
                    if isinstance(val, list):
                        bits.extend([str(x) for x in val])
                    elif val is not None:
                        bits.append(str(val))
                chunk = " ".join(x for x in bits if x).strip()
                if chunk:
                    text_chunks.append(chunk)
            for tag in soup.find_all(True):
                for attr in ("href", "src", "action", "data-href", "data-url"):
                    val = tag.get(attr)
                    if val:
                        links_blob.append(str(val))
            raw_html = str(soup)
            if raw_html:
                text_chunks.append(raw_html[:25000])
        except Exception:
            pass
    text_blob = "\n".join(x for x in text_chunks if x).lower()
    link_blob_l = "\n".join(links_blob).lower()

    detected: set[str] = set()
    link_matches: Dict[str, bool] = {}
    for provider, pats in oauth_patterns.items():
        found = any(re.search(pat, text_blob, flags=re.IGNORECASE) is not None for pat in pats)
        link_hit = any(dom in link_blob_l for dom in oauth_link_domains.get(provider, ()))
        if found or link_hit:
            detected.add(provider)
        link_matches[provider] = bool(link_hit)
    return {
        "oauth_providers_detected": sorted(detected),
        "oauth_provider_link_matches": link_matches,
    }


def _enrich_capture_and_html_signals(
    *,
    input_url: str,
    capture_json: Optional[Dict[str, Any]],
    soup: Any,
    html_structure_summary: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[str]]:
    """Return (layer2_enrichment, layer3_enrichment, strong_signals, weak_signals)."""
    cj = capture_json or {}
    security_block = _detect_security_block_page(cj)
    final_url = str(cj.get("final_url") or input_url or "")
    input_reg = _reg_domain(input_url)
    final_reg = _reg_domain(final_url)
    chain = [str(x) for x in (cj.get("redirect_chain") or []) if x]
    chain_regs = [_reg_domain(x) for x in chain if _reg_domain(x)]
    cross_domain_redirect_count = int(cj.get("cross_domain_redirect_count") or 0)
    final_domain_is_free_hosting = any(final_reg.endswith(suf) for suf in _FREE_HOSTING_SUFFIXES)

    hs = html_structure_summary or {}
    dom = html_dom_summary or {}
    brand_terms = [str(x).lower() for x in (hs.get("brand_terms_found_in_text") or [])]
    brand_terms_present = bool(brand_terms)
    redirect_domain_mismatch = bool(
        len(set(chain_regs)) >= 2 and final_reg and any(r and r != final_reg for r in chain_regs)
    )

    # URL weak/contextual signals.
    parsed = urlparse(input_url if "://" in input_url else ("https://" + input_url))
    host = (parsed.hostname or "").lower()
    path_q = ((parsed.path or "") + "?" + (parsed.query or "")).lower()
    contains_punycode = "xn--" in input_url.lower()
    contains_non_ascii = any(ord(ch) > 127 for ch in input_url)
    encoded_char_count = len(re.findall(r"%[0-9a-fA-F]{2}", input_url))
    suspicious_keyword_count = sum(1 for k in _URL_WEAK_KEYWORDS if k in path_q)
    excessive_hyphen_count = host.count("-")
    brand_in_subdomain_or_path_but_not_registered_domain = bool(
        brand_terms and final_reg and any((b in host or b in path_q) and (b not in final_reg) for b in brand_terms)
    )

    html_capture_length = 0
    html_capture_missing_reason: Optional[str] = None
    script_tag_count = 0
    iframe_count = 0
    hidden_input_count = 0
    external_script_domain_count = 0
    external_form_action_count = int(dom.get("form_action_external_domain_count") or 0)
    suspicious_html_keyword_count = 0
    form_action_domain_mismatch = bool(external_form_action_count > 0)
    password_input_count = int(hs.get("password_input_count") or 0)
    password_input_external_action = bool(password_input_count > 0 and external_form_action_count > 0)
    sparse_login_like_layout = bool(dom.get("sparse_credential_capture_layout") or hs.get("sparse_login_like_layout"))
    missing_org_elements = bool(dom.get("missing_real_ecosystem_context") or (not hs.get("has_support_help_links") and hs.get("footer_link_count", 0) == 0))

    if soup is None:
        html_capture_missing_reason = "html_not_available"
    else:
        try:
            html_str = str(soup)
            html_capture_length = len(html_str)
            low = html_str.lower()
            script_tag_count = len(soup.find_all("script"))
            iframe_count = len(soup.find_all("iframe"))
            hidden_input_count = sum(1 for i in soup.find_all("input") if str((i.get("type") or "")).lower() == "hidden")
            script_domains = set()
            for s in soup.find_all("script", src=True):
                src = str(s.get("src") or "")
                d = _reg_domain(src)
                if d and d != final_reg:
                    script_domains.add(d)
            external_script_domain_count = len(script_domains)
            suspicious_html_keyword_count = sum(1 for k in _SUSPICIOUS_HTML_KEYWORDS if k in low)
        except Exception:
            html_capture_missing_reason = "html_parse_partial"

    final_path = ""
    try:
        final_path = ((urlparse(final_url).path or "") + "?" + (urlparse(final_url).query or "")).lower()
    except Exception:
        final_path = ""
    page_family = str(dom.get("page_family") or "").strip().lower()
    form_count = int(hs.get("form_count") or 0)
    email_inputs = int(hs.get("email_input_count") or 0)
    phone_inputs = int(hs.get("phone_input_count") or 0)
    credential_capture = password_input_count > 0 or (form_count > 0 and (email_inputs > 0 or phone_inputs > 0))
    cross_domain_credential = bool(external_form_action_count > 0 and (password_input_count > 0 or form_count > 0))
    auth_path = any(t in (final_path or path_q) for t in ("login", "signin", "auth", "verify", "account", "password", "recover"))
    auth_payment_path = any(t in (final_path or path_q) for t in ("login", "signin", "auth", "verify", "password", "recover", "checkout", "payment", "billing", "pay"))
    official_platform_candidate = bool(host_on_official_brand_apex(_hostname(final_url)) or str(dom.get("official_platform_context")) == "true")
    host_path_brand_context = bool(brand_in_subdomain_or_path_but_not_registered_domain or auth_path)
    creator_platform_candidate = bool(final_reg in _CREATOR_PLATFORM_SUFFIXES and _hostname(final_url).endswith("." + final_reg))
    creator_404_or_newsletter = bool(
        creator_platform_candidate
        and any(x in f"{str(hs.get('title') or '').lower()}\n{str(hs.get('visible_text_snippet') or '').lower()}\n{final_path}" for x in ("404", "not found", "page not found", "newsletter", "/404"))
    )
    creator_major_brand_surface = bool(
        _contains_major_brand_token(_hostname(final_url))
        or _contains_major_brand_token(final_path)
        or _contains_major_brand_token(str(hs.get("title") or ""))
        or _contains_major_brand_token(str(hs.get("h1_text") or ""))
    )
    trust_surface_blob = (
        str(hs.get("title") or "")
        + " "
        + str(hs.get("h1_text") or "")
        + " "
        + str(hs.get("visible_text_snippet") or "")
    ).lower()
    auth_terms_present = any(
        t in trust_surface_blob
        for t in ("login", "sign in", "signin", "verify", "account", "password", "recover", "checkout", "secure")
    ) or bool(re.search(r"\b(payment|billing|pay|pricing)\b", trust_surface_blob))
    brand_in_trust_surface_text = any(b in trust_surface_blob for b in brand_terms)
    brand_auth_surface_context = bool(brand_in_trust_surface_text and auth_terms_present)
    auth_payment_brand_context = bool(
        # Credentials/forms alone are not sufficient; mismatched brand must be part of trust/auth context.
        (
            page_family in {"auth_login_recovery", "checkout_payment"}
            or auth_payment_path
            or credential_capture
            or cross_domain_credential
        )
        and (brand_auth_surface_context or host_path_brand_context or bool(dom.get("trust_surface_brand_domain_mismatch")))
    )
    trust_surface_brand_context = bool(
        dom.get("trust_surface_brand_domain_mismatch")
        or dom.get("title_brand_domain_mismatch")
        or dom.get("h1_brand_domain_mismatch")
        or dom.get("strong_branding_without_official_domain")
        or dom.get("logo_domain_mismatch")
        or int(dom.get("anchor_strong_mismatch_count") or 0) > 0
    )
    content_rich_profile = bool(dom.get("content_rich_profile") or page_family in {"article_news", "content_feed_forum_aggregator", "public_docs_or_reference", "generic_landing"})
    no_cred_capture = bool((password_input_count == 0) and (not cross_domain_credential) and int(dom.get("form_action_external_domain_count") or 0) == 0)
    brand_in_domain_or_path_context = bool(
        brand_terms_present
        and final_reg
        and any((b in _hostname(final_url) or b in final_path) and (b not in final_reg) for b in brand_terms)
    )
    form_action_targets = dom.get("form_action_targets_summary") or []
    brand_in_form_action_context = False
    if isinstance(form_action_targets, list):
        for row in form_action_targets:
            if not isinstance(row, dict):
                continue
            ad = str(row.get("action_domain") or "").lower()
            if ad and any((b in ad) and (b not in final_reg) for b in brand_terms):
                brand_in_form_action_context = True
                break
    brand_in_auth_payment_context = bool(brand_terms_present and (auth_payment_brand_context or brand_auth_surface_context))
    brand_domain_mismatch = bool(
        brand_terms_present
        and (
            brand_in_domain_or_path_context
            or brand_in_form_action_context
            or (credential_capture and (brand_in_domain_or_path_context or trust_surface_brand_context or brand_in_auth_payment_context))
            or brand_in_auth_payment_context
        )
    )
    official_auth_same_domain_candidate = bool(
        input_reg
        and final_reg
        and input_reg == final_reg
        and official_platform_candidate
        and int(dom.get("form_action_external_domain_count") or 0) == 0
        and not bool(password_input_external_action)
        and not bool(final_domain_is_free_hosting)
    )
    if official_auth_same_domain_candidate and not brand_in_domain_or_path_context and not brand_in_form_action_context:
        # OAuth/provider mentions on first-party official authwalls should not create host-brand mismatch.
        brand_domain_mismatch = False
    suppression_same_domain_content_rich = bool(
        input_reg
        and final_reg
        and input_reg == final_reg
        and int(hs.get("password_input_count") or 0) == 0
        and int(hs.get("email_input_count") or 0) == 0
        and int(dom.get("form_action_external_domain_count") or 0) == 0
        and bool(dom.get("content_rich_profile") or (int(hs.get("nav_link_count") or 0) >= 4 and int(hs.get("footer_link_count") or 0) >= 2))
    )
    if suppression_same_domain_content_rich and not brand_in_domain_or_path_context and not brand_in_form_action_context and not brand_in_auth_payment_context:
        brand_domain_mismatch = False
    creator_ok, creator_brand, creator_reasons = _creator_platform_host_candidate(
        final_host=_hostname(final_url),
        final_reg=final_reg,
        title=str(hs.get("title") or ""),
        visible_text=str(hs.get("visible_text_snippet") or ""),
    )
    resource_only_brand_context = bool(
        int(dom.get("branded_resource_domains_non_official") or 0) > 0
        and not trust_surface_brand_context
        and not host_path_brand_context
        and not auth_payment_brand_context
    )
    incidental_content_brand_context = bool(
        content_rich_profile
        and no_cred_capture
        and bool(brand_terms)
        and not trust_surface_brand_context
        and not host_path_brand_context
        and not auth_payment_brand_context
    )

    def _classify_brand_domain_mismatch_strength() -> str:
        if not brand_domain_mismatch:
            return "none"
        if (
            creator_ok
            and no_cred_capture
            and not cross_domain_credential
            and not creator_major_brand_surface
            and not trust_surface_brand_context
            and not auth_payment_brand_context
        ):
            # Creator-platform entity-consistent pages can include social/payment/platform ecosystem brand mentions.
            return "weak"
        if (
            creator_404_or_newsletter
            and no_cred_capture
            and not cross_domain_credential
            and not creator_major_brand_surface
            and not trust_surface_brand_context
            and not auth_payment_brand_context
        ):
            return "weak"
        if official_platform_candidate and not trust_surface_brand_context and not auth_payment_brand_context:
            # Official platform pages can mention many brands contextually (UTM, customer logos, OAuth labels, marketing copy).
            return "weak"
        if creator_ok and no_cred_capture and not trust_surface_brand_context and not host_path_brand_context:
            return "weak"
        if host_path_brand_context or trust_surface_brand_context or auth_payment_brand_context:
            return "strong"
        if final_domain_is_free_hosting and bool(brand_in_subdomain_or_path_but_not_registered_domain):
            return "strong"
        if resource_only_brand_context or incidental_content_brand_context:
            return "weak"
        return "strong"

    brand_mismatch_strength = _classify_brand_domain_mismatch_strength()
    coherence = compute_brand_domain_coherence(
        registrable_domain=final_reg,
        title=str(hs.get("title") or ""),
        h1=str(hs.get("h1_text") or ""),
        visible_text_sample=str(hs.get("visible_text_snippet") or ""),
    )
    layer2 = {
        "input_registered_domain": input_reg,
        "final_registered_domain": final_reg,
        "redirect_chain_registered_domains": chain_regs,
        "cross_domain_redirect_count": cross_domain_redirect_count,
        "brand_domain_mismatch": brand_domain_mismatch,
        "brand_domain_mismatch_strength": brand_mismatch_strength,
        "host_path_brand_context": host_path_brand_context,
        "trust_surface_brand_context": trust_surface_brand_context,
        "auth_payment_brand_context": auth_payment_brand_context,
        "resource_only_brand_context": resource_only_brand_context,
        "incidental_content_brand_context": incidental_content_brand_context,
        "brand_mismatch_suppressed_same_domain_content_rich": suppression_same_domain_content_rich,
        "brand_domain_coherence_score": float(coherence.get("brand_domain_coherence_score") or 0.0),
        "brand_domain_coherence_match": bool(coherence.get("brand_domain_coherence_match")),
        "brand_domain_coherence_reason": str(coherence.get("brand_domain_coherence_reason") or ""),
        "domain_brand_tokens": list(coherence.get("domain_brand_tokens") or []),
        "page_brand_candidates": list(coherence.get("page_brand_candidates") or []),
        "platform_hosted_brand_consistency_candidate": creator_ok,
        "platform_hosted_brand_consistency_reasons": creator_reasons,
        "platform_hosted_brand": creator_brand or None,
        "redirect_domain_mismatch": redirect_domain_mismatch,
        "final_domain_is_free_hosting": final_domain_is_free_hosting,
        "contains_punycode": contains_punycode,
        "contains_non_ascii": contains_non_ascii,
        "encoded_char_count": encoded_char_count,
        "suspicious_keyword_count": suspicious_keyword_count,
        "excessive_hyphen_count": excessive_hyphen_count,
        "brand_in_subdomain_or_path_but_not_registered_domain": brand_in_subdomain_or_path_but_not_registered_domain,
        "weak_signal_note": "URL anomalies are contextual/weak by themselves.",
        "security_block_page_detected": bool(security_block.get("security_block_page_detected")),
        "security_block_page_vendor": security_block.get("security_block_page_vendor"),
        "security_block_page_reasons": list(security_block.get("security_block_page_reasons") or []),
        "uses_https": bool(cj.get("uses_https")),
        "browser_security_state": str(cj.get("browser_security_state") or "") or None,
        "tls_or_cert_error_detected": bool(cj.get("tls_or_cert_error_detected")),
        "insecure_scheme_detected": bool(cj.get("insecure_scheme_detected")),
        "mixed_content_detected": bool(cj.get("mixed_content_detected")),
        "security_state_reasons": list(cj.get("security_state_reasons") or []),
    }
    click_meta = ((cj.get("interaction") or {}) if isinstance(cj.get("interaction"), dict) else {})
    layer2["click_probe"] = {
        k: v
        for k, v in click_meta.items()
        if isinstance(k, str) and k.startswith("click_probe")
    }
    language_enrichment = _fasttext_language_enrichment(cj, soup)
    oauth_enrichment = _detect_oauth_providers(cj, soup)
    layer3 = {
        "script_tag_count": script_tag_count,
        "iframe_count": iframe_count,
        "hidden_input_count": hidden_input_count,
        "external_script_domain_count": external_script_domain_count,
        "external_form_action_count": external_form_action_count,
        "suspicious_html_keyword_count": suspicious_html_keyword_count,
        "html_capture_length": html_capture_length,
        "html_capture_missing_reason": html_capture_missing_reason,
        "form_action_domain_mismatch": form_action_domain_mismatch,
        "password_input_external_action": password_input_external_action,
        "sparse_login_like_layout": sparse_login_like_layout,
        "missing_org_elements": missing_org_elements,
        "detected_language": language_enrichment.get("detected_language"),
        "detected_language_confidence": language_enrichment.get("detected_language_confidence"),
        "language_detection_available": language_enrichment.get("language_detection_available"),
        "language_detection_error": language_enrichment.get("language_detection_error"),
        "language_mismatch_contextual_signal": language_enrichment.get("language_mismatch_contextual_signal"),
        "oauth_providers_detected": oauth_enrichment.get("oauth_providers_detected") or [],
        "oauth_provider_link_matches": oauth_enrichment.get("oauth_provider_link_matches") or {},
    }

    strong: List[str] = []
    weak: List[str] = []
    if brand_domain_mismatch and brand_mismatch_strength == "strong":
        strong.append("Brand-to-final-domain mismatch detected.")
    elif brand_domain_mismatch and brand_mismatch_strength == "weak":
        weak.append("Brand mention appears contextual on non-auth/content page; mismatch treated as informational.")
    if redirect_domain_mismatch:
        strong.append("Redirect chain contains domain transitions inconsistent with final domain.")
    if final_domain_is_free_hosting:
        strong.append("Final domain appears on free-hosting suffix list.")
    if form_action_domain_mismatch:
        strong.append("Form action posts to external domain.")
    if password_input_external_action:
        strong.append("Password input exists with external form action.")
    if sparse_login_like_layout:
        strong.append("Sparse login-like layout detected.")
    if missing_org_elements:
        strong.append("Expected support/privacy/footer context is weak or missing.")

    if contains_punycode:
        weak.append("URL contains punycode label (contextual).")
    if contains_non_ascii:
        weak.append("URL contains non-ASCII characters (contextual).")
    if encoded_char_count > 0:
        weak.append("URL has encoded characters (contextual).")
    if suspicious_keyword_count > 0:
        weak.append("URL contains auth/payment keywords (contextual).")
    if excessive_hyphen_count >= 4:
        weak.append("Hostname has many hyphens (contextual).")
    if brand_in_subdomain_or_path_but_not_registered_domain:
        weak.append("Brand appears in subdomain/path but not registered domain (contextual).")
    if layer3.get("language_mismatch_contextual_signal"):
        weak.append("Detected page language differs from baseline language signal (contextual only).")
    if layer3.get("language_detection_available"):
        weak.append(
            "Detected page language may help identify suspicious localization mismatches, "
            "but multilingual legitimate sites are common, so this is treated as contextual evidence only."
        )
    return layer2, layer3, strong, weak


def _classify_capture_failure_type(error_msg: str) -> str:
    e = (error_msg or "").lower()
    if "err_network_access_denied" in e or "network_access_denied" in e:
        return "network_access_denied"
    if "timeout" in e or "timed out" in e:
        return "timeout"
    dns_tokens = (
        "err_name_not_resolved",
        "err_connection_refused",
        "err_connection_reset",
        "err_connection_closed",
        "err_connection_timed_out",
        "err_connection",
        "err_tunnel",
        "err_cert",
        "err_ssl",
    )
    if any(t in e for t in dns_tokens):
        return "dns_or_connection"
    if "net::" in e or "page.goto" in e or "goto:" in e or "navigation" in e:
        return "browser_navigation_error"
    return "unknown_failure"


def _capture_failure_plain_reason(failure_type: str) -> str:
    gap = (
        " HTML was unavailable because live capture failed; missing live evidence should be treated "
        "as an evidence gap, not as proof of legitimacy."
    )
    if failure_type == "network_access_denied":
        return (
            "Live capture failed with network access denied; evasive or blocked pages can limit "
            "automated phishing analysis."
            + gap
        )
    if failure_type == "timeout":
        return (
            "Live capture timed out; this may be benign, but it limits live-page and HTML validation." + gap
        )
    if failure_type == "dns_or_connection":
        return (
            "Live capture failed due to DNS or TLS/connection issues; automated HTML and DOM checks "
            "could not run on the final page."
            + gap
        )
    if failure_type == "browser_navigation_error":
        return (
            "Live capture failed during browser navigation; automated HTML and DOM checks may be incomplete."
            + gap
        )
    return (
        "Live capture did not complete successfully; treat missing live evidence as an analysis gap, not legitimacy."
        + gap
    )


def _merge_capture_failure_fields(cap: Dict[str, Any], ml: Dict[str, Any]) -> None:
    """Populate additive capture-failure suspicion fields on the reinforcement capture dict (in place)."""
    err = str(cap.get("error") or "").strip()
    strategy = str(cap.get("capture_strategy") or "")
    capture_failed = bool(err) or strategy == "failed"

    p_raw = ml.get("phish_proba_model_raw")
    if p_raw is None and ml.get("phish_proba") is not None:
        p_raw = ml.get("phish_proba")
    p_cal = ml.get("phish_proba_calibrated", p_raw)
    try:
        p_cal_f = float(p_cal) if p_cal is not None else 0.0
    except (TypeError, ValueError):
        p_cal_f = 0.0

    if not capture_failed:
        cap.update(
            {
                "capture_failed": False,
                "capture_failure_type": None,
                "capture_failure_suspicious": False,
                "capture_failure_suspicion_level": "none",
                "capture_failure_reason": None,
            }
        )
    else:
        ftype = _classify_capture_failure_type(err)
        level = "moderate" if p_cal_f >= 0.70 else "weak"
        cap.update(
            {
                "capture_failed": True,
                "capture_failure_type": ftype,
                "capture_failure_suspicious": True,
                "capture_failure_suspicion_level": level,
                "capture_failure_reason": _capture_failure_plain_reason(ftype),
            }
        )

    html_p = str(cap.get("html_path") or "").strip()
    has_html = bool(html_p) and Path(html_p).is_file()
    if capture_failed:
        cap["capture_status_display"] = "failed"
    elif not has_html or bool(cap.get("capture_blocked")) or strategy == "http_fallback":
        cap["capture_status_display"] = "partial"
    else:
        cap["capture_status_display"] = "success"


def _finalize_click_probe_diagnostics_on_capture(
    cap: Dict[str, Any],
    *,
    enable_click_probe: bool,
) -> None:
    """Normalize click-probe diagnostics when capture ends before or without the probe."""
    cp = dict(cap.get("click_probe") or {}) if isinstance(cap.get("click_probe"), dict) else {}
    failed = bool(cap.get("capture_failed"))
    if not enable_click_probe:
        cp["click_probe_enabled"] = False
        if not cp.get("click_probe_skip_reason"):
            cp["click_probe_skip_reason"] = "disabled_by_config"
        cap["click_probe"] = cp
        return
    cp["click_probe_enabled"] = True
    if failed and not cp.get("click_probe_skip_reason"):
        cp["click_probe_skip_reason"] = "capture_failed_before_probe"
        cp["click_probe_attempted"] = False
        if not cp.get("click_probe_candidate_texts_sample"):
            cp["click_probe_candidate_texts_sample"] = []
        cp.setdefault("click_probe_candidate_count", 0)
    cap["click_probe"] = cp
