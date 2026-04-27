"""Layer-1 ML + optional reinforcement capture → dashboard JSON (no screenshot-vs-reference flow).

The optional **official-brand apex cap** below is a *temporary UX safety net* only. Primary trust
should be the retrained Layer-1 model and structural brand features in
:mod:`src.pipeline.features.brand_signals`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import tldextract

from src.pipeline.features.brand_signals import BRAND_TOKENS
from src.pipeline.features.brand_signals import host_on_official_brand_apex
from src.pipeline.paths import analysis_dir, ensure_layout
from src.pipeline.safe_url import safe_hostname

from .ai_adjudicator import run_ai_adjudication
from .capture import capture_url
from .config import PipelineConfig
from .html_dom_anomaly_signals import extract_html_dom_anomaly_signals
from .html_structure_signals import extract_html_structure_signals
from .host_path_reasoning import (
    assess_host_path_reasoning,
    blend_ml_phish_for_host_path_reasoning,
)
from .legitimacy_bundle import (
    adjust_org_risk_for_legitimacy,
    blend_ml_phish_for_legitimacy,
    build_legitimacy_bundle,
)
from .ml_layer1 import predict_layer1
from .org_style_signals import dampen_org_style_for_page_family, org_style_from_capture_blob
from .schemas import utc_now_iso
from .utils.download_models import resolve_fasttext_model_path
from .verdict_policy import Verdict3WayConfig, verdict_3way

logger = logging.getLogger(__name__)

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
_URL_WEAK_KEYWORDS = ("login", "verify", "secure", "account", "update", "payment", "auth", "wallet")
_SUSPICIOUS_HTML_KEYWORDS = ("verify", "password", "account", "security", "urgent", "suspend", "billing")
_FASTTEXT_MIN_TEXT_CHARS = 120
_FASTTEXT_MODEL_CACHE: Any = None
_FASTTEXT_MODEL_ERROR: Optional[str] = None
_TRUSTED_DOMAIN_REGISTRY_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_PLATFORM_DOMAIN_REGISTRY_CACHE: Optional[List[Dict[str, Any]]] = None
_REPO_ROOT = Path(__file__).resolve().parents[2]
_IMPERSONATION_BRAND_TOKENS = set(BRAND_TOKENS) | {"handshake"}


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


def _load_trusted_domain_registry(csv_path: str) -> Dict[str, Dict[str, Any]]:
    global _TRUSTED_DOMAIN_REGISTRY_CACHE
    if _TRUSTED_DOMAIN_REGISTRY_CACHE is not None:
        return _TRUSTED_DOMAIN_REGISTRY_CACHE
    p = Path(csv_path)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    if not p.is_file():
        _TRUSTED_DOMAIN_REGISTRY_CACHE = {}
        return _TRUSTED_DOMAIN_REGISTRY_CACHE
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rd = str(row.get("registered_domain") or "").strip().lower()
                if not rd:
                    continue
                hosts_raw = str(row.get("allowed_hosts") or "").strip()
                allowed_hosts = {h.strip().lower() for h in hosts_raw.split("|") if h.strip()}
                org = str(row.get("organization") or "").strip()
                notes = str(row.get("notes") or "").strip()
                out[rd] = {
                    "registered_domain": rd,
                    "allowed_hosts": allowed_hosts,
                    "organization": org,
                    "expected_cname_contains": str(row.get("expected_cname_contains") or "").strip().lower(),
                    "expected_ns_contains": str(row.get("expected_ns_contains") or "").strip().lower(),
                    "expected_asn_org_contains": str(row.get("expected_asn_org_contains") or "").strip().lower(),
                    "notes": notes,
                }
    except Exception:
        out = {}
    _TRUSTED_DOMAIN_REGISTRY_CACHE = out
    return out


def _load_platform_domain_registry(csv_path: str) -> List[Dict[str, Any]]:
    global _PLATFORM_DOMAIN_REGISTRY_CACHE
    if _PLATFORM_DOMAIN_REGISTRY_CACHE is not None:
        return _PLATFORM_DOMAIN_REGISTRY_CACHE
    p = Path(csv_path)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    if not p.is_file():
        _PLATFORM_DOMAIN_REGISTRY_CACHE = []
        return _PLATFORM_DOMAIN_REGISTRY_CACHE
    rows: List[Dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = str(row.get("platform_name") or "").strip()
                if not name:
                    continue
                def _split(col: str) -> set[str]:
                    raw = str(row.get(col) or "").strip()
                    return {x.strip().lower() for x in raw.split("|") if x.strip()}
                rows.append(
                    {
                        "platform_name": name,
                        "official_registered_domains": _split("official_registered_domains"),
                        "official_hosts": _split("official_hosts"),
                        "user_hosting_registered_domains": _split("user_hosting_registered_domains"),
                        "allowed_oauth_providers": _split("allowed_oauth_providers"),
                        "notes": str(row.get("notes") or "").strip(),
                    }
                )
    except Exception:
        rows = []
    _PLATFORM_DOMAIN_REGISTRY_CACHE = rows
    return rows


def _classify_platform_host_context(
    *,
    layer2_capture: Optional[Dict[str, Any]],
    html_dom_enrichment: Optional[Dict[str, Any]],
    host_path_reasoning: Optional[Dict[str, Any]],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    cap = layer2_capture or {}
    enrich = html_dom_enrichment or {}
    hp = host_path_reasoning or {}
    final_url = str(cap.get("final_url") or "")
    final_host = _hostname(final_url)
    final_reg = str(cap.get("final_registered_domain") or "").lower()
    path_l = (urlparse(final_url).path or "").lower() if final_url else ""
    text = f"{str(cap.get('title') or '').lower()}\n{str(cap.get('visible_text_sample') or '').lower()}"
    rows = _load_platform_domain_registry(getattr(cfg, "platform_domains_csv_path", ""))

    oauth_found: set[str] = {
        str(x).strip().lower()
        for x in (enrich.get("oauth_providers_detected") or [])
        if str(x).strip()
    }
    if not oauth_found:
        oauth_aliases: Dict[str, Tuple[str, ...]] = {
            "google": ("login with google", "continue with google", "sign in with google"),
            "facebook": ("login with facebook", "continue with facebook", "sign in with facebook"),
            "github": ("login with github", "continue with github", "sign in with github"),
            "apple": ("login with apple", "continue with apple", "sign in with apple"),
            "square": ("login with square", "continue with square", "sign in with square"),
        }
        for provider, phrases in oauth_aliases.items():
            if any(p in text for p in phrases):
                oauth_found.add(provider)
    loginish_path = any(tok in path_l for tok in ("login", "signin", "sign-in", "auth", "recovery", "account", "dashboard"))
    inactive_markers = ("404", "not found", "site unavailable", "project not published", "page not found")
    is_inactive_page = any(m in text for m in inactive_markers)
    reasons: List[str] = []
    blockers: List[str] = []
    platform_name: Optional[str] = None
    context_type = "unknown"

    matched_entry: Optional[Dict[str, Any]] = None
    for row in rows:
        if final_reg in set(row.get("official_registered_domains") or set()) or final_reg in set(
            row.get("user_hosting_registered_domains") or set()
        ):
            matched_entry = row
            break
    if matched_entry is None:
        return {
            "platform_context_type": context_type,
            "platform_name": platform_name,
            "platform_context_reasons": reasons,
            "oauth_providers_detected": sorted(oauth_found),
            "platform_context_blockers": blockers,
        }

    platform_name = str(matched_entry.get("platform_name") or "")
    official_regs = set(matched_entry.get("official_registered_domains") or set())
    official_hosts = set(matched_entry.get("official_hosts") or set())
    user_regs = set(matched_entry.get("user_hosting_registered_domains") or set())
    allowed_oauth = set(matched_entry.get("allowed_oauth_providers") or set())

    if final_reg in official_regs:
        is_explicit_official_host = final_host in official_hosts
        safe_official_subdomain = _safe_subdomain_allowed(final_host, final_reg) and str(
            hp.get("host_identity_class") or ""
        ) != "suspicious_host_pattern"
        if is_explicit_official_host or (safe_official_subdomain and final_reg not in user_regs):
            context_type = "official_platform_domain"
            reasons.append("Final host/domain matches official platform infrastructure.")
            allowed_detected = sorted([p for p in oauth_found if p in allowed_oauth])
            if loginish_path and allowed_detected:
                context_type = "official_platform_login"
                reasons.append("Official platform login/auth context with expected OAuth providers.")
        elif final_reg in user_regs:
            context_type = "user_hosted_subdomain"
            reasons.append("Registrable domain supports user-hosted pages; host is not an official platform host.")
        else:
            reasons.append("Official registrable domain matched but host did not match expected official hosts.")
    elif final_reg in user_regs:
        context_type = "user_hosted_subdomain"
        reasons.append("Final registrable domain is user-hosting infrastructure.")

    if context_type == "user_hosted_subdomain":
        cloud_imp = _detect_cloud_hosted_brand_impersonation(
            final_registered_domain=final_reg,
            final_host=final_host,
            final_url=final_url,
        )
        if cloud_imp:
            context_type = "cloud_hosted_brand_impersonation"
            blockers.append("cloud_hosted_brand_impersonation")
            reasons.append("Brand-like token on user-hosted cloud/builder infrastructure.")
        elif is_inactive_page and str(hp.get("host_identity_class") or "") == "suspicious_host_pattern":
            blockers.append("dormant_phishing_infra")
            reasons.append("Inactive page on suspicious user-hosted infrastructure.")

    return {
        "platform_context_type": context_type,
        "platform_name": platform_name,
        "platform_context_reasons": reasons,
        "oauth_providers_detected": sorted(oauth_found),
        "oauth_provider_link_matches": enrich.get("oauth_provider_link_matches") or {},
        "platform_context_blockers": blockers,
    }


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
    brand_domain_mismatch = bool(brand_terms and final_reg and not any(b in final_reg for b in brand_terms))
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

    def _classify_brand_domain_mismatch_strength() -> str:
        if not brand_domain_mismatch:
            return "none"
        final_path = ""
        try:
            final_path = ((urlparse(final_url).path or "") + "?" + (urlparse(final_url).query or "")).lower()
        except Exception:
            final_path = ""
        page_family = str(dom.get("page_family") or "").strip().lower()
        auth_path = any(t in (final_path or path_q) for t in ("login", "signin", "auth", "verify", "account", "password", "recover"))
        auth_context = page_family == "auth_login_recovery" or auth_path
        form_count = int(hs.get("form_count") or 0)
        email_inputs = int(hs.get("email_input_count") or 0)
        phone_inputs = int(hs.get("phone_input_count") or 0)
        credential_capture = password_input_count > 0 or (form_count > 0 and (email_inputs > 0 or phone_inputs > 0))
        payment_brand = any(b in {"paypal", "stripe"} for b in brand_terms)
        payment_flow = any(t in (final_path or path_q) for t in ("checkout", "payment", "billing", "pay")) or bool(
            "checkout_payment" == page_family
        )
        payment_impersonation = bool(payment_brand and (payment_flow or form_count > 0))
        cross_domain_credential = bool(external_form_action_count > 0 and (password_input_count > 0 or form_count > 0))
        if auth_context or credential_capture or payment_impersonation or cross_domain_credential:
            return "strong"
        contentish_path = any(t in (final_path or path_q) for t in ("pricing", "blog", "product", "features", "integrations", "docs"))
        contextual_mentions = any(
            t in ((str(hs.get("visible_text_snippet") or "") + " " + str(hs.get("title") or "")).lower())
            for t in ("pay with", "supports", "integration", "integrations", "accepted payment", "payment methods")
        )
        if contentish_path or contextual_mentions:
            return "weak"
        return "strong"

    brand_mismatch_strength = _classify_brand_domain_mismatch_strength()
    layer2 = {
        "input_registered_domain": input_reg,
        "final_registered_domain": final_reg,
        "redirect_chain_registered_domains": chain_regs,
        "cross_domain_redirect_count": cross_domain_redirect_count,
        "brand_domain_mismatch": brand_domain_mismatch,
        "brand_domain_mismatch_strength": brand_mismatch_strength,
        "redirect_domain_mismatch": redirect_domain_mismatch,
        "final_domain_is_free_hosting": final_domain_is_free_hosting,
        "contains_punycode": contains_punycode,
        "contains_non_ascii": contains_non_ascii,
        "encoded_char_count": encoded_char_count,
        "suspicious_keyword_count": suspicious_keyword_count,
        "excessive_hyphen_count": excessive_hyphen_count,
        "brand_in_subdomain_or_path_but_not_registered_domain": brand_in_subdomain_or_path_but_not_registered_domain,
        "weak_signal_note": "URL anomalies are contextual/weak by themselves.",
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


_ML_CAPTURE_MISS_SAFETY_REASON_UNCERTAIN = (
    "Deterministic safety guardrail applied: ML predicted phishing, live capture failed, and no trusted-domain anchor "
    "was present; result kept uncertain instead of likely legitimate."
)
_ML_CAPTURE_MISS_SAFETY_REASON_HIGH = (
    "Deterministic safety guardrail applied: ML predicted phishing, live capture failed, and no trusted-domain anchor "
    "was present; result was not kept likely legitimate given high calibrated phishing risk."
)


def _trust_anchor_present(bundle: Dict[str, Any]) -> bool:
    return bool(
        bundle.get("official_registrable_anchor")
        or bundle.get("official_domain_family")
        or bundle.get("strong_trust_anchor")
    )


def _html_missing_after_capture_failure(
    *,
    capture_failed: bool,
    html_capture_missing_reason: Optional[str],
    html_structure_error: Optional[str],
    html_dom_anomaly_error: Optional[str],
) -> bool:
    """True when live/HTML evidence is missing in ways consistent with a failed or incomplete capture."""
    if not capture_failed:
        return False
    if html_capture_missing_reason == "html_not_available":
        return True
    if html_structure_error in ("missing_html_path", "html_path_not_found"):
        return True
    if html_dom_anomaly_error == "missing_html_path":
        return True
    return False


def _apply_ml_phishing_capture_miss_legitimacy_safety(
    verdict: Dict[str, Any],
    *,
    ml: Dict[str, Any],
    legitimacy_bundle: Dict[str, Any],
    capture_failed: bool,
    html_capture_missing_reason: Optional[str],
    html_structure_error: Optional[str],
    html_dom_anomaly_error: Optional[str],
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    """Global invariant: ML predicts phishing + capture/HTML gap + no trust anchor → never likely_legitimate."""
    out = dict(verdict)
    if not bool(ml.get("predicted_phishing")):
        return out
    if _trust_anchor_present(legitimacy_bundle):
        return out
    if not capture_failed:
        return out
    if not _html_missing_after_capture_failure(
        capture_failed=True,
        html_capture_missing_reason=html_capture_missing_reason,
        html_structure_error=html_structure_error,
        html_dom_anomaly_error=html_dom_anomaly_error,
    ):
        return out
    v = str(out.get("verdict_3way") or out.get("label") or "")
    if v != "likely_legitimate":
        return out

    p_raw = ml.get("phish_proba_model_raw")
    if p_raw is None and ml.get("phish_proba") is not None:
        p_raw = ml.get("phish_proba")
    p_cal = ml.get("phish_proba_calibrated", p_raw)
    try:
        pcf = float(p_cal) if p_cal is not None else 0.0
    except (TypeError, ValueError):
        pcf = 0.0

    vcfg = verdict_cfg or Verdict3WayConfig()
    lo, hi = float(vcfg.combined_low), float(vcfg.combined_high)
    uncertain_interior = min(hi - 1e-3, max(lo + 1e-3, (lo + hi) / 2.0))

    prev_cs = out.get("combined_score")
    if out.get("combined_score_pre_ml_capture_miss_safety") is None:
        out["combined_score_pre_ml_capture_miss_safety"] = (
            float(prev_cs) if isinstance(prev_cs, (int, float)) else None
        )
    out["ml_phishing_capture_miss_safety_applied"] = True
    reasons = list(out.get("reasons") or [])
    out["reasons"] = reasons

    if pcf >= 0.70:
        cs = float(prev_cs) if isinstance(prev_cs, (int, float)) else 0.0
        out["combined_score"] = max(cs, hi)
        v2, why = verdict_3way(float(out["combined_score"]), vcfg)
        out["label"] = v2
        out["verdict_3way"] = v2
        out["confidence"] = "medium" if v2 != "uncertain" else "low"
        out["post_rescue_rule"] = why
        if _ML_CAPTURE_MISS_SAFETY_REASON_HIGH not in reasons:
            reasons.append(_ML_CAPTURE_MISS_SAFETY_REASON_HIGH)
        out["reasons"] = reasons
        return out

    out["combined_score"] = uncertain_interior
    out["label"] = "uncertain"
    out["verdict_3way"] = "uncertain"
    out["confidence"] = "medium" if pcf >= 0.60 else "low"
    out["post_rescue_rule"] = f"ml_capture_miss_safety_uncertain (calibrated_p_phish~{pcf:.3f})"
    if _ML_CAPTURE_MISS_SAFETY_REASON_UNCERTAIN not in reasons:
        reasons.append(_ML_CAPTURE_MISS_SAFETY_REASON_UNCERTAIN)
    out["reasons"] = reasons
    return out


def _apply_capture_failure_verdict_hardening(
    verdict: Dict[str, Any],
    *,
    ml: Dict[str, Any],
    legitimacy_bundle: Dict[str, Any],
    capture_failed: bool,
    html_missing_for_reinforcement: bool,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    """Optional post-policy nudge: do not let incomplete live evidence alone collapse a high-ML case to uncertain."""
    cfg = verdict_cfg or Verdict3WayConfig()
    out = dict(verdict)
    if bool(out.get("untrusted_builder_hosting_signal_applied")):
        # Respect conservative builder-host downgrade when already applied.
        return out
    out.setdefault("capture_failure_verdict_guardrail_applied", False)
    strong_legit = bool(
        legitimacy_bundle.get("official_registrable_anchor")
        or legitimacy_bundle.get("official_domain_family")
        or legitimacy_bundle.get("strong_trust_anchor")
    )
    p_raw = ml.get("phish_proba_model_raw")
    if p_raw is None and ml.get("phish_proba") is not None:
        p_raw = ml.get("phish_proba")
    p_cal = ml.get("phish_proba_calibrated", p_raw)
    try:
        p_cal_f = float(p_cal) if p_cal is not None else 0.0
    except (TypeError, ValueError):
        p_cal_f = 0.0
    pred = bool(ml.get("predicted_phishing"))
    vlabel = str(out.get("label") or "")
    cs = out.get("combined_score")

    out["capture_failure_high_ml_incomplete_warning"] = None
    if capture_failed and html_missing_for_reinforcement and p_cal_f >= 0.70 and not strong_legit:
        out["capture_failure_high_ml_incomplete_warning"] = (
            "High ML phishing probability with failed live capture — treat as suspicious despite incomplete reinforcement."
        )

    if not (
        capture_failed
        and html_missing_for_reinforcement
        and p_cal_f >= 0.70
        and pred
        and not strong_legit
        and vlabel == "uncertain"
        and isinstance(cs, (int, float))
    ):
        return out

    bumped = max(float(cs), cfg.combined_high)
    out["combined_score_pre_capture_failure_guardrail"] = float(cs)
    out["combined_score"] = bumped
    out["capture_failure_verdict_guardrail_applied"] = True
    v2, vwhy = verdict_3way(bumped, cfg)
    out["label"] = v2
    out["verdict_3way"] = v2
    out["confidence"] = "medium" if v2 != "uncertain" else "low"
    out["post_rescue_rule"] = vwhy
    out["reasons"] = list(out.get("reasons") or []) + [
        "Guardrail: high calibrated phishing probability with failed capture and missing HTML prevented "
        "an uncertain-only outcome without live reinforcement."
    ]
    return out


def _apply_official_brand_apex_cap(ml: Dict[str, Any], url: str) -> Dict[str, Any]:
    """Temporary safety net — lower *displayed* P(phish) on official-brand host families (not primary classifier)."""
    if ml.get("error") or ml.get("phish_proba") is None:
        return ml
    canon = (ml.get("canonical_url") or url or "").strip()
    h, _ = safe_hostname(canon)
    if not host_on_official_brand_apex(h):
        return ml
    raw_display = float(ml["phish_proba"])
    if raw_display <= _OFFICIAL_BRAND_APEX_PHISH_CAP:
        return ml
    capped = round(min(raw_display, _OFFICIAL_BRAND_APEX_PHISH_CAP), 6)
    out = {
        **ml,
        "phish_proba": capped,
        "phish_proba_pre_apex_cap": raw_display,
        "official_brand_apex_cap_applied": True,
    }
    if out.get("phish_proba_model_raw") is None:
        out["phish_proba_model_raw"] = raw_display
    out["predicted_phishing"] = bool(capped >= 0.5)
    return out


def _verdict_from_scores(
    phish_ml_effective: Optional[float],
    phish_proba_model_raw: Optional[float],
    phish_proba_calibrated: Optional[float],
    org_risk_raw: float,
    org_risk_adjusted: float,
    *,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    if phish_ml_effective is None:
        return {
            "label": "uncertain",
            "confidence": "low",
            "reasons": ["ML model unavailable; rely on reinforcement signals only."],
        }
    combined = min(1.0, max(0.0, 0.65 * float(phish_ml_effective) + 0.35 * float(org_risk_adjusted)))
    combined_no_reinforcement = min(1.0, max(0.0, 0.65 * float(phish_ml_effective)))
    reinforcement_combined_delta = float(combined - combined_no_reinforcement)
    org_adjustment_delta = float(org_risk_adjusted - org_risk_raw)
    vlabel, vwhy = verdict_3way(combined, verdict_cfg or Verdict3WayConfig())
    conf = "medium" if vlabel != "uncertain" else "low"
    pr = phish_proba_model_raw
    pc = phish_proba_calibrated
    reasons: List[str] = [
        f"Layer-1 phishing probability (raw model) ~ {pr:.3f}." if pr is not None else "Layer-1 raw probability n/a.",
        (
            f"Layer-1 calibrated probability ~ {pc:.3f}."
            if pc is not None and pr is not None and abs(pc - pr) > 1e-6
            else "Layer-1 calibration unchanged or unavailable."
        ),
        f"Effective ML score after legitimacy blend ~ {float(phish_ml_effective):.3f}.",
        f"Org-style reinforcement (raw) ~ {org_risk_raw:.3f}; adjusted ~ {org_risk_adjusted:.3f}.",
        vwhy,
    ]
    return {
        "label": vlabel,
        "verdict_3way": vlabel,
        "confidence": conf,
        "combined_score": combined,
        "combined_score_without_reinforcement": combined_no_reinforcement,
        "reinforcement_combined_delta": reinforcement_combined_delta,
        "org_risk_raw": org_risk_raw,
        "org_risk_adjusted": org_risk_adjusted,
        "org_adjustment_delta": org_adjustment_delta,
        "reasons": reasons,
    }


def _apply_legitimacy_rescue_on_verdict(
    verdict: Dict[str, Any],
    *,
    ml: Dict[str, Any],
    host_path_reasoning: Optional[Dict[str, Any]],
    html_structure_summary: Optional[Dict[str, Any]],
    html_structure_risk: Optional[float],
    html_dom_summary: Optional[Dict[str, Any]],
    html_dom_risk: Optional[float],
    layer2_capture: Optional[Dict[str, Any]],
    legitimacy_bundle: Dict[str, Any],
    cfg: PipelineConfig,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    """Generalized legitimacy rescue: downgrade high-ML verdicts when strong structural legitimacy exists."""
    out = dict(verdict)
    out.setdefault("legitimacy_rescue_applied", False)
    out.setdefault("legitimacy_rescue_reasons", [])
    out.setdefault("legitimacy_rescue_blockers", [])
    out.setdefault("legitimacy_strong_override_triggered", False)
    out.setdefault("legitimacy_strong_override_conditions", [])
    out["legitimacy_rescue_original_ml_score"] = ml.get("phish_proba")

    if not bool(getattr(cfg, "legitimacy_rescue_enabled", True)):
        return out
    c0 = out.get("combined_score")
    if not isinstance(c0, (int, float)):
        return out
    v0 = str(out.get("verdict_3way") or out.get("label") or "")
    if v0 != "likely_phishing":
        return out
    ml_score = ml.get("phish_proba")
    if not isinstance(ml_score, (int, float)) or float(ml_score) < 0.65:
        return out

    hp = host_path_reasoning or {}
    hs = html_structure_summary or {}
    dom = html_dom_summary or {}
    cap = layer2_capture or {}
    final_reg = str(cap.get("final_registered_domain") or "")
    final_url = str(cap.get("final_url") or "")
    final_host = _hostname(final_url)
    cloud_imp = _detect_cloud_hosted_brand_impersonation(
        final_registered_domain=final_reg,
        final_host=final_host,
        final_url=final_url,
    )
    blockers = _compute_phishing_blockers(
        host_path_reasoning=hp,
        html_dom_summary=dom,
        layer2_capture=cap,
        legitimacy_bundle=legitimacy_bundle,
        cloud_hosted_brand_impersonation=cloud_imp,
    )
    if blockers:
        out["legitimacy_rescue_blockers"] = blockers
        return out

    input_reg = str(cap.get("input_registered_domain") or "")
    redirect_regs = [str(x) for x in (cap.get("redirect_chain_registered_domains") or []) if str(x)]
    redirect_set = {x for x in redirect_regs if x}
    same_domain_redirects = (not redirect_set) or (redirect_set <= {input_reg, final_reg})
    no_cross_domain_redirect = int(cap.get("cross_domain_redirect_count") or 0) == 0
    same_domain_form_actions = int(dom.get("form_action_external_domain_count") or 0) == 0
    html_structure_ok = (
        float(html_structure_risk) if isinstance(html_structure_risk, (int, float)) else 0.0
    ) <= float(getattr(cfg, "legitimacy_rescue_max_html_structure_risk", 0.32))
    html_dom_ok = (
        float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else 0.0
    ) <= float(getattr(cfg, "legitimacy_rescue_max_html_dom_anomaly_risk", 0.30))
    no_brand_mismatch = not _is_strong_brand_domain_mismatch(cap, dom)
    host_legit = str(hp.get("host_legitimacy_confidence") or "") in {"high", "medium"}
    path_plaus = str(hp.get("path_fit_assessment") or "") == "plausible"

    trusted_registry = _load_trusted_domain_registry(getattr(cfg, "trusted_domains_csv_path", ""))
    trusted_entry = trusted_registry.get(final_reg)
    trusted_support = False
    if trusted_entry:
        allowed_hosts = set(trusted_entry.get("allowed_hosts") or set())
        trusted_support = (not allowed_hosts and bool(final_reg)) or final_host in allowed_hosts or final_host.endswith("." + final_reg) or final_host == final_reg

    rescue_reasons: List[str] = []
    if input_reg and final_reg and input_reg == final_reg:
        rescue_reasons.append("input_final_registered_domain_match")
    if same_domain_redirects and no_cross_domain_redirect:
        rescue_reasons.append("redirect_chain_domain_consistent")
    if same_domain_form_actions:
        rescue_reasons.append("same_org_form_action_targets")
    if html_structure_ok and html_dom_ok:
        rescue_reasons.append("low_structural_risk")
    if host_legit and path_plaus:
        rescue_reasons.append("host_path_reasonable_for_login_or_account_flow")
    if trusted_support:
        rescue_reasons.append("trusted_domain_registry_support")

    strong_override_conditions: List[str] = []
    if input_reg and final_reg and input_reg == final_reg:
        strong_override_conditions.append("input_final_registered_domain_match")
    if same_domain_redirects and no_cross_domain_redirect:
        strong_override_conditions.append("same_domain_redirect_chain")
    if same_domain_form_actions:
        strong_override_conditions.append("no_cross_domain_form_action")
    hs_risk_v = float(html_structure_risk) if isinstance(html_structure_risk, (int, float)) else None
    if hs_risk_v is not None and abs(hs_risk_v) <= 1e-9:
        strong_override_conditions.append("html_structure_risk_zero")
    dom_risk_v = float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else None
    if dom_risk_v is not None and dom_risk_v <= 0.2:
        strong_override_conditions.append("html_dom_anomaly_risk_le_0p2")
    if no_brand_mismatch:
        strong_override_conditions.append("brand_domain_match")
    out["legitimacy_strong_override_conditions"] = strong_override_conditions

    strong_override = len(strong_override_conditions) == 6
    if strong_override:
        out["legitimacy_strong_override_triggered"] = True
        cap_score = float(getattr(cfg, "legitimacy_rescue_ml_cap_after_rescue", 0.52))
        cap_score = min(max(cap_score, 0.45), 0.55)
        vcfg = verdict_cfg or Verdict3WayConfig()
        lo, hi = float(vcfg.combined_low), float(vcfg.combined_high)
        uncertain_mid = min(hi - 1e-3, max(lo + 1e-3, (lo + hi) / 2.0))
        c1_force = min(float(c0), min(cap_score, uncertain_mid))
        out["combined_score_pre_legitimacy_rescue"] = float(c0)
        out["combined_score"] = c1_force
        out["effective_ml_score_pre_legitimacy_rescue"] = ml.get("phish_proba")
        out["effective_ml_score_post_legitimacy_rescue"] = cap_score
        out["legitimacy_rescue_applied"] = True
        out["legitimacy_rescue_adjustment"] = float(c1_force - float(c0))
        out["legitimacy_rescue_ml_cap_applied"] = cap_score
        out["legitimacy_rescue_reasons"] = rescue_reasons + ["strong_legitimacy_override_tier"]
        out["legitimacy_rescue_target_verdict"] = "uncertain"
        out["label"] = "uncertain"
        out["verdict_3way"] = "uncertain"
        out["confidence"] = "medium"
        out["reasons"] = list(out.get("reasons") or []) + [
            "Legitimacy rescue applied: same-domain redirects/forms and low structural risk reduced confidence in ML-only phishing prediction."
        ]
        return out

    min_required = 4 if trusted_support else 5
    if len(rescue_reasons) < min_required or not no_brand_mismatch:
        out["legitimacy_rescue_reasons"] = rescue_reasons
        if not no_brand_mismatch:
            out["legitimacy_rescue_blockers"] = ["brand_domain_mismatch"]
        return out

    cap_score = float(getattr(cfg, "legitimacy_rescue_ml_cap_after_rescue", 0.52))
    cap_score = min(max(cap_score, 0.45), 0.55)
    vcfg = verdict_cfg or Verdict3WayConfig()
    lo, hi = float(vcfg.combined_low), float(vcfg.combined_high)
    uncertain_mid = min(hi - 1e-3, max(lo + 1e-3, (lo + hi) / 2.0))
    target_verdict = str(getattr(cfg, "legitimacy_rescue_target_verdict", "uncertain") or "uncertain").lower()
    if target_verdict == "likely_legitimate":
        target_score = min(lo - 1e-3, cap_score)
    elif target_verdict == "likely_phishing":
        target_score = max(hi, cap_score)
    else:
        target_score = min(cap_score, uncertain_mid)
        target_verdict = "uncertain"
    c1 = min(float(c0), float(target_score))

    out["combined_score_pre_legitimacy_rescue"] = float(c0)
    out["combined_score"] = c1
    out["effective_ml_score_pre_legitimacy_rescue"] = ml.get("phish_proba")
    out["effective_ml_score_post_legitimacy_rescue"] = min(float(ml_score), cap_score)
    out["legitimacy_rescue_applied"] = True
    out["legitimacy_rescue_adjustment"] = float(c1 - float(c0))
    out["legitimacy_rescue_ml_cap_applied"] = cap_score
    out["legitimacy_rescue_reasons"] = rescue_reasons
    out["legitimacy_rescue_target_verdict"] = target_verdict
    out["reasons"] = list(out.get("reasons") or []) + [
        "Legitimacy rescue applied: same-domain redirects/forms and low structural risk reduced confidence in ML-only phishing prediction."
    ]
    if target_verdict == "uncertain":
        out["label"] = "uncertain"
        out["verdict_3way"] = "uncertain"
        out["confidence"] = "medium"
    else:
        v2, vwhy = verdict_3way(c1, vcfg)
        out["label"] = v2
        out["verdict_3way"] = v2
        out["confidence"] = "medium" if v2 != "uncertain" else "low"
        out["post_rescue_rule"] = vwhy
    return out


def _dns_contribution_profile(ml: Dict[str, Any]) -> Dict[str, Any]:
    tops = ml.get("top_linear_signals") or []
    if not isinstance(tops, list) or not tops:
        return {"dns_contribution_share": None, "dns_features_detected": [], "dns_dominant": False}
    total = 0.0
    dns_sum = 0.0
    dns_feats: List[str] = []
    for row in tops:
        if not isinstance(row, dict):
            continue
        coef = row.get("signed_coef")
        val = row.get("value")
        if not isinstance(coef, (int, float)) or not isinstance(val, (int, float)):
            continue
        contrib = abs(float(coef) * float(val))
        total += contrib
        feat = str(row.get("feature") or "")
        if "dns" in feat.lower():
            dns_sum += contrib
            dns_feats.append(feat)
    share = (dns_sum / total) if total > 1e-12 else 0.0
    return {
        "dns_contribution_share": round(float(share), 4),
        "dns_features_detected": sorted(set(dns_feats)),
        "dns_dominant": False,
    }


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


def _compute_phishing_blockers(
    *,
    host_path_reasoning: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    layer2_capture: Optional[Dict[str, Any]],
    legitimacy_bundle: Dict[str, Any],
    cloud_hosted_brand_impersonation: bool,
    suppress_oauth_brand_mismatch: bool = False,
    platform_context_type: str = "unknown",
) -> List[str]:
    hp = host_path_reasoning or {}
    dom = html_dom_summary or {}
    cap = layer2_capture or {}
    mismatch_strength = str(cap.get("brand_domain_mismatch_strength") or "").lower()
    strong_brand_mismatch = bool(
        mismatch_strength == "strong"
        and (bool(cap.get("brand_domain_mismatch")) or bool(dom.get("trust_surface_brand_domain_mismatch")))
    )
    blockers: List[str] = []
    if int(dom.get("form_action_external_domain_count") or 0) > 0 or bool(legitimacy_bundle.get("suspicious_form_action_cross_origin")):
        blockers.append("cross_domain_form_action")
    if strong_brand_mismatch and (not suppress_oauth_brand_mismatch):
        blockers.append("brand_domain_mismatch")
    if not bool(legitimacy_bundle.get("no_free_hosting_signal", True)) or bool(cap.get("final_domain_is_free_hosting")):
        blockers.append("free_hosting_impersonation")
    if bool(cap.get("contains_punycode")) or bool(cap.get("contains_non_ascii")):
        blockers.append("punycode_or_non_ascii_host")
    if cloud_hosted_brand_impersonation:
        blockers.append("cloud_hosted_brand_impersonation")
    if platform_context_type == "cloud_hosted_brand_impersonation":
        blockers.append("cloud_hosted_brand_impersonation")
    if bool(dom.get("suspicious_credential_collection_pattern")) or bool(dom.get("login_harvester_pattern")):
        blockers.append("sparse_credential_harvester_pattern")
    if bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern")):
        blockers.append("wrapper_or_interstitial_redirect_pattern")
    if int(dom.get("anchor_strong_mismatch_count") or 0) > 0:
        blockers.append("strong_anchor_domain_mismatch")
    if str(hp.get("host_identity_class") or "") == "suspicious_host_pattern":
        blockers.append("suspicious_host_pattern")
    return blockers


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
    strength = str(cap.get("brand_domain_mismatch_strength") or "").lower()
    return bool(
        strength == "strong"
        and (bool(cap.get("brand_domain_mismatch")) or bool(dom.get("trust_surface_brand_domain_mismatch")))
    )


def _evaluate_hosting_domain_trust(
    *,
    layer2_capture: Optional[Dict[str, Any]],
    html_structure_risk: Optional[float],
    html_dom_risk: Optional[float],
    blockers: List[str],
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    cap = layer2_capture or {}
    input_reg = str(cap.get("input_registered_domain") or "").lower()
    final_reg = str(cap.get("final_registered_domain") or "").lower()
    final_url = str(cap.get("final_url") or "")
    final_host = _hostname(final_url)
    reg = _load_trusted_domain_registry(getattr(cfg, "trusted_domains_csv_path", ""))
    entry = reg.get(final_reg)

    reasons: List[str] = []
    mismatches: List[str] = []
    evidence: Dict[str, Any] = {
        "input_registered_domain": input_reg,
        "final_registered_domain": final_reg,
        "final_host": final_host,
    }
    identity_match = bool(input_reg and final_reg and input_reg == final_reg)
    identity_registry_match = bool(entry is not None and final_reg == str(entry.get("registered_domain") or ""))
    host_allowed = False
    if identity_registry_match:
        allowed_hosts = set(entry.get("allowed_hosts") or set())
        if allowed_hosts:
            host_allowed = final_host in allowed_hosts
        else:
            host_allowed = _safe_subdomain_allowed(final_host, final_reg)
    evidence["identity_match"] = identity_match
    evidence["identity_registry_match"] = identity_registry_match
    evidence["host_allowed"] = host_allowed

    cloud_imp = _detect_cloud_hosted_brand_impersonation(
        final_registered_domain=final_reg,
        final_host=final_host,
        final_url=final_url,
    )
    evidence["cloud_hosted_brand_impersonation"] = cloud_imp
    if cloud_imp:
        mismatches.append("cloud_hosted_brand_impersonation")

    if not identity_match:
        reasons.append("Identity check failed: input and final registrable domains differ.")
        return {
            "hosting_trust_status": "hosting_trust_unknown",
            "hosting_trust_reasons": reasons,
            "hosting_trust_evidence": evidence,
            "hosting_trust_mismatches": mismatches,
            "cloud_hosted_brand_impersonation": cloud_imp,
            "trust_promotion_eligible": False,
        }

    if entry is None:
        reasons.append("No trusted registry entry for final registrable domain.")
        return {
            "hosting_trust_status": "hosting_trust_unknown",
            "hosting_trust_reasons": reasons,
            "hosting_trust_evidence": evidence,
            "hosting_trust_mismatches": mismatches,
            "cloud_hosted_brand_impersonation": cloud_imp,
            "trust_promotion_eligible": False,
        }

    if not host_allowed:
        mismatches.append("host_not_allowed_by_registry")
        reasons.append("Final host does not match allowed hosts for registry entry.")
        return {
            "hosting_trust_status": "hosting_trust_mismatch",
            "hosting_trust_reasons": reasons,
            "hosting_trust_evidence": evidence,
            "hosting_trust_mismatches": mismatches,
            "cloud_hosted_brand_impersonation": cloud_imp,
            "trust_promotion_eligible": False,
        }

    cname_obs = str(cap.get("dns_cname") or "").lower()
    ns_obs = str(cap.get("dns_ns") or "").lower()
    asn_obs = str(cap.get("dns_asn_org") or "").lower()
    expected_cname = str(entry.get("expected_cname_contains") or "").lower()
    expected_ns = str(entry.get("expected_ns_contains") or "").lower()
    expected_asn = str(entry.get("expected_asn_org_contains") or "").lower()
    evidence.update(
        {
            "observed_cname": cname_obs or None,
            "observed_ns": ns_obs or None,
            "observed_asn_org": asn_obs or None,
            "expected_cname_contains": expected_cname or None,
            "expected_ns_contains": expected_ns or None,
            "expected_asn_org_contains": expected_asn or None,
        }
    )
    checks: List[Tuple[str, str, str]] = [
        ("cname", expected_cname, cname_obs),
        ("ns", expected_ns, ns_obs),
        ("asn", expected_asn, asn_obs),
    ]
    matched = 0
    expected_count = 0
    for name, exp, obs in checks:
        if not exp:
            continue
        expected_count += 1
        if exp in obs:
            matched += 1
        elif obs:
            mismatches.append(f"{name}_mismatch")

    hs_ok = (float(html_structure_risk) if isinstance(html_structure_risk, (int, float)) else 1.0) <= float(
        getattr(cfg, "legitimacy_rescue_max_html_structure_risk", 0.32)
    )
    hd_ok = (float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else 1.0) <= float(
        getattr(cfg, "legitimacy_rescue_max_html_dom_anomaly_risk", 0.30)
    )
    no_blockers = not blockers and not cloud_imp

    if expected_count == 0:
        status = "hosting_trust_partial"
        reasons.append("Identity and host validation passed; DNS/ASN expectations unavailable.")
    elif matched == expected_count:
        status = "hosting_trust_verified"
        reasons.append("Identity passed and expected DNS/ASN hosting signatures matched.")
    elif matched > 0:
        status = "hosting_trust_partial"
        reasons.append("Identity passed with partial DNS/ASN hosting signature match.")
    else:
        status = "hosting_trust_mismatch"
        reasons.append("Identity passed but DNS/ASN hosting signatures did not match expected values.")

    promotion_eligible = status in {"hosting_trust_verified", "hosting_trust_partial"} and no_blockers and hs_ok and hd_ok
    return {
        "hosting_trust_status": status,
        "hosting_trust_reasons": reasons,
        "hosting_trust_evidence": evidence,
        "hosting_trust_mismatches": mismatches,
        "cloud_hosted_brand_impersonation": cloud_imp,
        "trust_promotion_eligible": bool(promotion_eligible),
    }


def _apply_hosting_trust_promotion(
    verdict: Dict[str, Any],
    *,
    trust: Dict[str, Any],
    ml: Dict[str, Any],
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    out = dict(verdict)
    out["hosting_trust_status"] = trust.get("hosting_trust_status", "hosting_trust_unknown")
    out["hosting_trust_reasons"] = list(trust.get("hosting_trust_reasons") or [])
    out["hosting_trust_evidence"] = trust.get("hosting_trust_evidence") or {}
    out["hosting_trust_mismatches"] = list(trust.get("hosting_trust_mismatches") or [])
    out["cloud_hosted_brand_impersonation"] = bool(trust.get("cloud_hosted_brand_impersonation"))
    out["hosting_trust_promotion_applied"] = False

    if out["cloud_hosted_brand_impersonation"]:
        out["reasons"] = list(out.get("reasons") or []) + [
            "Cloud-hosted brand impersonation pattern detected; legitimacy promotion disabled."
        ]
        return out

    if not bool(trust.get("trust_promotion_eligible")):
        if out.get("hosting_trust_status") == "hosting_trust_mismatch":
            out["reasons"] = list(out.get("reasons") or []) + [
                "Hosting trust mismatch observed despite domain identity check; adding suspicion context."
            ]
        return out

    v0 = str(out.get("verdict_3way") or out.get("label") or "")
    if v0 != "uncertain":
        return out
    mlp = ml.get("phish_proba")
    if isinstance(mlp, (int, float)) and float(mlp) >= 0.82:
        # Strong ML phishing still requires additional evidence beyond trust support.
        return out
    cs = out.get("combined_score")
    if not isinstance(cs, (int, float)):
        return out
    vcfg = verdict_cfg or Verdict3WayConfig()
    lo = float(vcfg.combined_low)
    promoted_score = min(lo - 1e-3, float(cs))
    out["combined_score_pre_hosting_trust_promotion"] = float(cs)
    out["combined_score"] = promoted_score
    out["label"] = "likely_legitimate"
    out["verdict_3way"] = "likely_legitimate"
    out["confidence"] = "medium"
    out["hosting_trust_promotion_applied"] = True
    out["reasons"] = list(out.get("reasons") or []) + [
        "Domain identity and hosting trust signals align with low structural risk; uncertain verdict promoted conservatively."
    ]
    return out


def _apply_untrusted_builder_hosting_downgrade(
    verdict: Dict[str, Any],
    *,
    layer2_capture: Optional[Dict[str, Any]],
    html_structure_summary: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    html_dom_enrichment: Optional[Dict[str, Any]],
    blockers: List[str],
) -> Dict[str, Any]:
    """Downgrade likely_phishing to uncertain only for inactive/non-threat builder-hosted pages."""
    out = dict(verdict)
    out.setdefault("untrusted_builder_hosting_signal_applied", False)
    cap = layer2_capture or {}
    hs = html_structure_summary or {}
    dom = html_dom_summary or {}
    enrich = html_dom_enrichment or {}
    v0 = str(out.get("verdict_3way") or out.get("label") or "")
    if v0 != "likely_phishing":
        return out

    final_reg = str(cap.get("final_registered_domain") or "").lower()
    final_url = str(cap.get("final_url") or "")
    final_host = _hostname(final_url)
    if final_reg not in _BUILDER_PLATFORM_SUFFIXES:
        return out

    has_brand_impersonation = _is_strong_brand_domain_mismatch(cap, dom)
    has_credential_harvest = bool(dom.get("suspicious_credential_collection_pattern")) or bool(dom.get("login_harvester_pattern"))
    has_cross_domain_form = int(dom.get("form_action_external_domain_count") or 0) > 0
    if has_brand_impersonation or has_credential_harvest or has_cross_domain_form:
        return out
    hard_blockers = list(blockers or [])
    if hard_blockers:
        return out

    # Generic/untrusted hostname heuristic: non-root subdomain on builder platform.
    generic_untrusted_hostname = bool(final_host and final_host != final_reg and final_host.endswith("." + final_reg))
    if not generic_untrusted_hostname:
        return out

    capture_failed = bool(cap.get("capture_failed"))
    html_missing_reason = str(enrich.get("html_capture_missing_reason") or "").strip().lower()
    if capture_failed or html_missing_reason in {"html_not_available", "html_parse_partial"}:
        # No downgrade on missing evidence; keep conservative phishing stance.
        return out

    # Positive non-threat evidence required: clear inactive/not-found marker + no forms/auth language.
    title = str(cap.get("title") or "").lower()
    body = str(cap.get("visible_text_sample") or "").lower()
    text = f"{title}\n{body}"
    inactive_markers = (
        "404",
        "not found",
        "site unavailable",
        "site not found",
        "project not published",
        "this site does not exist",
        "page not found",
    )
    has_inactive_marker = any(m in text for m in inactive_markers)
    form_count = int(hs.get("form_count") or 0)
    password_inputs = int(hs.get("password_input_count") or 0)
    auth_terms = ("login", "signin", "sign in", "verify", "verification", "payment", "billing", "password", "account")
    auth_language_present = any(t in text for t in auth_terms) or int(enrich.get("suspicious_html_keyword_count") or 0) > 0
    if not has_inactive_marker:
        return out
    if form_count > 0 or password_inputs > 0 or auth_language_present:
        return out

    out["untrusted_builder_hosting_signal_applied"] = True
    out["untrusted_builder_hosting_reason"] = "Untrusted hosting platform with generic hostname; legitimacy cannot be confirmed."
    out["combined_score_pre_untrusted_builder_downgrade"] = out.get("combined_score")
    out["label"] = "uncertain"
    out["verdict_3way"] = "uncertain"
    out["confidence"] = "low"
    cs = out.get("combined_score")
    if isinstance(cs, (int, float)):
        out["combined_score"] = min(float(cs), 0.55)
    out["reasons"] = list(out.get("reasons") or []) + [out["untrusted_builder_hosting_reason"]]
    return out


def _apply_inactive_site_overlay(
    verdict: Dict[str, Any],
    *,
    ml: Dict[str, Any],
    layer2_capture: Optional[Dict[str, Any]],
    html_structure_summary: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    html_dom_enrichment: Optional[Dict[str, Any]],
    blockers: List[str],
) -> Dict[str, Any]:
    """Mark low-evidence inactive pages as uncertain without implying legitimacy."""
    out = dict(verdict)
    out.setdefault("inactive_site_detected", False)
    cap = layer2_capture or {}
    hs = html_structure_summary or {}
    dom = html_dom_summary or {}
    enrich = html_dom_enrichment or {}
    text = f"{str(cap.get('title') or '').lower()}\n{str(cap.get('visible_text_sample') or '').lower()}"

    has_capture_failure = bool(cap.get("capture_failed"))
    if bool(out.get("dormant_phishing_infra_detected")):
        return out
    inactive_markers = (
        "404",
        "not found",
        "site unavailable",
        "site not found",
        "project not published",
        "this site does not exist",
        "page not found",
    )
    has_inactive_page_marker = any(marker in text for marker in inactive_markers)
    if not (has_capture_failure or has_inactive_page_marker):
        return out

    form_count = int(hs.get("form_count") or 0)
    password_inputs = int(hs.get("password_input_count") or 0)
    cross_domain_form = int(dom.get("form_action_external_domain_count") or 0) > 0
    has_brand_impersonation = _is_strong_brand_domain_mismatch(cap, dom)
    has_credential_harvest = bool(dom.get("suspicious_credential_collection_pattern")) or bool(dom.get("login_harvester_pattern"))
    auth_terms = ("login", "signin", "sign in", "auth", "verify", "verification", "payment", "billing", "password", "account")
    has_auth_language = any(term in text for term in auth_terms) or int(enrich.get("suspicious_html_keyword_count") or 0) > 0

    high_conf_phishing = False
    try:
        p_cal = ml.get("phish_proba_calibrated")
        p_raw = ml.get("phish_proba")
        p = float(p_cal if p_cal is not None else p_raw)
        high_conf_phishing = p >= 0.80
    except Exception:
        high_conf_phishing = False
    high_conf_phishing = high_conf_phishing or bool(ml.get("predicted_phishing") and str(out.get("verdict_3way") or out.get("label") or "") == "likely_phishing")

    strong_blockers = set(blockers or [])
    if (
        cross_domain_form
        or has_auth_language
        or has_brand_impersonation
        or has_credential_harvest
        or "suspicious_host_pattern" in strong_blockers
        or high_conf_phishing
    ):
        return out

    html_missing_reason = str(enrich.get("html_capture_missing_reason") or "").strip().lower()
    no_html_content = html_missing_reason in {"html_not_available", "html_parse_partial"}
    minimal_placeholder = has_inactive_page_marker and len((cap.get("visible_text_sample") or "").strip()) <= 280
    if not (no_html_content or minimal_placeholder):
        return out
    if form_count > 0 or password_inputs > 0:
        return out

    out["inactive_site_detected"] = True
    out["inactive_site_label"] = "inactive / no live content available"
    out["inactive_site_explanation"] = "Content unavailable - classification limited"
    out["combined_score_pre_inactive_site_overlay"] = out.get("combined_score")
    out["label"] = "uncertain"
    out["verdict_3way"] = "uncertain"
    out["confidence"] = "low"
    out["reasons"] = list(out.get("reasons") or []) + [
        "Page could not be validated due to missing or inactive content."
    ]
    return out


def _apply_platform_context_policy(
    verdict: Dict[str, Any],
    *,
    platform_context: Dict[str, Any],
    html_structure_summary: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    html_structure_risk: Optional[float],
    html_dom_risk: Optional[float],
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    out = dict(verdict)
    ptype = str(platform_context.get("platform_context_type") or "unknown")
    pre = out.get("combined_score")
    out["platform_context_type"] = ptype
    out["platform_name"] = platform_context.get("platform_name")
    out["platform_context_reasons"] = list(platform_context.get("platform_context_reasons") or [])
    out["oauth_providers_detected"] = list(platform_context.get("oauth_providers_detected") or [])
    out["oauth_provider_link_matches"] = dict(platform_context.get("oauth_provider_link_matches") or {})
    out["oauth_brand_mismatch_suppressed"] = bool(platform_context.get("oauth_brand_mismatch_suppressed"))
    out["dormant_phishing_infra_detected"] = bool(platform_context.get("dormant_phishing_infra_detected"))
    out["dormant_phishing_infra_reasons"] = list(platform_context.get("dormant_phishing_infra_reasons") or [])

    if ptype == "cloud_hosted_brand_impersonation":
        if isinstance(pre, (int, float)) and float(pre) < 0.72:
            out["combined_score_pre_platform_context_adjustment"] = float(pre)
            out["combined_score"] = 0.72
        out["label"] = "likely_phishing"
        out["verdict_3way"] = "likely_phishing"
        out["confidence"] = "medium"
        out["reasons"] = list(out.get("reasons") or []) + [
            "Cloud-hosted brand impersonation detected on user-hosted infrastructure."
        ]
        return out

    if bool(out.get("dormant_phishing_infra_detected")):
        if isinstance(pre, (int, float)) and float(pre) < 0.68:
            out["combined_score_pre_platform_context_adjustment"] = float(pre)
            out["combined_score"] = 0.68
        out["label"] = "likely_phishing"
        out["verdict_3way"] = "likely_phishing"
        out["confidence"] = "low"
        out["reasons"] = list(out.get("reasons") or []) + [
            "Inactive page on suspicious user-hosted infrastructure consistent with dormant phishing setup."
        ]
        return out

    if ptype == "official_platform_login":
        hs = html_structure_summary or {}
        dom = html_dom_summary or {}
        form_external = int(dom.get("form_action_external_domain_count") or 0) > 0
        no_blockers = not bool(platform_context.get("platform_context_blockers"))
        hs_ok = (float(html_structure_risk) if isinstance(html_structure_risk, (int, float)) else 1.0) <= 0.45
        hd_ok = (float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else 1.0) <= 0.45
        no_impersonation = not bool(dom.get("trust_surface_brand_domain_mismatch"))
        no_pw_external = not bool(int(hs.get("password_input_count") or 0) > 0 and form_external)
        if no_blockers and hs_ok and hd_ok and no_impersonation and no_pw_external:
            cs = out.get("combined_score")
            if isinstance(cs, (int, float)):
                if str(out.get("verdict_3way") or out.get("label") or "") == "likely_phishing":
                    out["combined_score_pre_platform_context_adjustment"] = float(cs)
                    out["combined_score"] = min(float(cs), 0.55)
                    out["label"] = "uncertain"
                    out["verdict_3way"] = "uncertain"
                    out["confidence"] = "low"
                    out["reasons"] = list(out.get("reasons") or []) + [
                        "Official platform login context with expected OAuth flow reduced false-positive risk."
                    ]
                else:
                    lo = float((verdict_cfg or Verdict3WayConfig()).combined_low)
                    if float(cs) <= 0.48:
                        out["combined_score_pre_platform_context_adjustment"] = float(cs)
                        out["combined_score"] = min(float(cs), lo - 1e-3)
                        out["label"] = "likely_legitimate"
                        out["verdict_3way"] = "likely_legitimate"
                        out["confidence"] = "medium"
                        out["reasons"] = list(out.get("reasons") or []) + [
                            "Official platform login context aligned with low structural risk."
                        ]
    return out


def _apply_dns_feature_dominance_dampening(
    *,
    ml_effective_score: Optional[float],
    ml: Dict[str, Any],
    layer2_capture: Optional[Dict[str, Any]],
    html_structure_risk: Optional[float],
    html_dom_risk: Optional[float],
    legitimacy_bundle: Dict[str, Any],
    cfg: PipelineConfig,
) -> Tuple[Optional[float], Dict[str, Any]]:
    meta = _dns_contribution_profile(ml)
    if ml_effective_score is None:
        return ml_effective_score, meta
    cap = layer2_capture or {}
    strong_legit = bool(
        str(cap.get("input_registered_domain") or "")
        and str(cap.get("final_registered_domain") or "")
        and str(cap.get("input_registered_domain") or "") == str(cap.get("final_registered_domain") or "")
        and int(cap.get("cross_domain_redirect_count") or 0) == 0
        and not _is_strong_brand_domain_mismatch(cap, None)
        and bool(legitimacy_bundle.get("no_deceptive_token_placement", True))
        and bool(legitimacy_bundle.get("no_free_hosting_signal", True))
        and (float(html_structure_risk) if isinstance(html_structure_risk, (int, float)) else 1.0)
        <= float(getattr(cfg, "legitimacy_rescue_max_html_structure_risk", 0.32))
        and (float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else 1.0)
        <= float(getattr(cfg, "legitimacy_rescue_max_html_dom_anomaly_risk", 0.30))
    )
    share = meta.get("dns_contribution_share")
    dominant = isinstance(share, (int, float)) and float(share) >= float(
        getattr(cfg, "legitimacy_rescue_dns_contribution_threshold", 0.35)
    )
    meta["dns_dominant"] = bool(dominant)
    meta["dns_dampening_applied"] = False
    meta["dns_dampening_factor"] = None
    meta["effective_ml_score_pre_dns_dampening"] = float(ml_effective_score)
    if not (strong_legit and dominant):
        meta["effective_ml_score_post_dns_dampening"] = float(ml_effective_score)
        return ml_effective_score, meta
    factor = min(max(float(getattr(cfg, "legitimacy_rescue_dns_dampening_factor", 0.78)), 0.5), 1.0)
    adjusted = float(ml_effective_score) * factor
    meta["dns_dampening_applied"] = True
    meta["dns_dampening_factor"] = factor
    meta["effective_ml_score_post_dns_dampening"] = float(adjusted)
    return adjusted, meta


def _apply_ml_overconfidence_cap(
    *,
    ml_effective_score: Optional[float],
    layer2_capture: Optional[Dict[str, Any]],
    html_structure_summary: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    html_structure_risk: Optional[float],
    html_dom_risk: Optional[float],
    host_path_reasoning: Optional[Dict[str, Any]],
    platform_context: Optional[Dict[str, Any]],
    hosting_trust: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "ml_overconfidence_cap_applied": False,
        "ml_overconfidence_cap_reason": None,
        "ml_score_before_overconfidence_cap": None,
        "ml_score_after_overconfidence_cap": None,
    }
    if ml_effective_score is None:
        return ml_effective_score, meta
    cap = layer2_capture or {}
    hs = html_structure_summary or {}
    dom = html_dom_summary or {}
    hp = host_path_reasoning or {}
    pctx = platform_context or {}
    trust = hosting_trust or {}
    s0 = float(ml_effective_score)
    meta["ml_score_before_overconfidence_cap"] = s0

    input_reg = str(cap.get("input_registered_domain") or "")
    final_reg = str(cap.get("final_registered_domain") or "")
    if not input_reg or not final_reg or input_reg != final_reg:
        return ml_effective_score, meta
    ptype = str(pctx.get("platform_context_type") or "unknown")
    if ptype in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}:
        return ml_effective_score, meta
    host_legit_high = str(hp.get("host_legitimacy_confidence") or "") == "high"
    trust_status = str(trust.get("hosting_trust_status") or "")
    official_or_trusted = ptype == "official_platform_domain" or host_legit_high or trust_status in {
        "hosting_trust_verified",
        "hosting_trust_partial",
    }
    if not official_or_trusted:
        return ml_effective_score, meta

    page_family = str(dom.get("page_family") or "")
    final_url = str(cap.get("final_url") or "")
    path_l = (urlparse(final_url).path or "").lower() if final_url else ""
    content_path = any(t in path_l for t in ("pricing", "blog", "product", "docs", "documentation", "feature", "integration"))
    content_family = page_family in {"article_news", "content_feed_forum_aggregator", "public_docs_or_reference", "generic_landing"}
    content_rich = bool(dom.get("content_rich_profile")) or content_family or content_path
    if not content_rich:
        return ml_effective_score, meta

    no_password = int(hs.get("password_input_count") or 0) == 0
    no_credential_harvest = not bool(dom.get("suspicious_credential_collection_pattern") or dom.get("login_harvester_pattern"))
    no_cross_domain_form = int(dom.get("form_action_external_domain_count") or 0) == 0
    no_cloud_imp = not bool(pctx.get("platform_context_type") == "cloud_hosted_brand_impersonation")
    no_suspicious_host = str(hp.get("host_identity_class") or "") != "suspicious_host_pattern"
    no_strong_mismatch = not _is_strong_brand_domain_mismatch(cap, dom)
    dom_ok = (float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else 1.0) <= 0.15
    hs_ok = (float(html_structure_risk) if isinstance(html_structure_risk, (int, float)) else 1.0) <= 0.30
    if not all((no_password, no_credential_harvest, no_cross_domain_form, no_cloud_imp, no_suspicious_host, no_strong_mismatch, dom_ok, hs_ok)):
        return ml_effective_score, meta

    capped = min(s0, 0.55)
    if capped >= s0 - 1e-9:
        meta["ml_score_after_overconfidence_cap"] = s0
        return ml_effective_score, meta
    meta["ml_overconfidence_cap_applied"] = True
    meta["ml_overconfidence_cap_reason"] = (
        "ML overconfidence capped due to strong official/content-page legitimacy evidence."
    )
    meta["ml_score_after_overconfidence_cap"] = float(capped)
    return float(capped), meta


def no_phishing_evidence_guard(
    *,
    html_structure_summary: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    html_dom_risk: Optional[float],
    host_path_reasoning: Optional[Dict[str, Any]],
    capture_failed: bool = False,
    html_structure_error: Optional[str] = None,
    html_capture_missing_reason: Optional[str] = None,
    ml_calibrated_phish: Optional[float] = None,
) -> bool:
    """Hard guard: if all major phishing indicators are absent, force legitimate."""
    if capture_failed:
        if ml_calibrated_phish is not None and float(ml_calibrated_phish) >= 0.70:
            return False
        if html_structure_error in {"missing_html_path", "html_path_not_found"}:
            return False
        if html_capture_missing_reason == "html_not_available":
            return False
    hs = html_structure_summary or {}
    dom = html_dom_summary or {}
    hp = host_path_reasoning or {}
    return bool(
        int(hs.get("password_input_count") or 0) == 0
        and int(dom.get("form_action_external_domain_count") or 0) == 0
        and not bool(dom.get("suspicious_credential_collection_pattern"))
        and not bool(dom.get("trust_action_context"))
        and not bool(dom.get("strong_impersonation_context"))
        and not bool(dom.get("wrapper_page_pattern"))
        and not bool(dom.get("login_harvester_pattern"))
        and float(html_dom_risk if isinstance(html_dom_risk, (int, float)) else 1.0) < 0.2
        and str(hp.get("host_legitimacy_confidence") or "") in {"medium", "high"}
        and str(hp.get("path_fit_assessment") or "") == "plausible"
    )


def _apply_no_phishing_evidence_override(
    verdict: Dict[str, Any],
    *,
    guard_triggered: bool,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    out = dict(verdict)
    out["no_phishing_evidence_guard"] = bool(guard_triggered)
    if not guard_triggered:
        return out
    prev = out.get("combined_score")
    out["combined_score_pre_no_phishing_evidence_override"] = float(prev) if isinstance(prev, (int, float)) else None
    c_new = min(0.35, float(prev) if isinstance(prev, (int, float)) else 0.35)
    out["combined_score"] = c_new
    out["legitimacy_rescue_applied"] = True
    out["legitimacy_rescue_adjustment"] = float(c_new - float(prev)) if isinstance(prev, (int, float)) else 0.0
    out["label"] = "likely_legitimate"
    out["verdict_3way"] = "likely_legitimate"
    out["confidence"] = "medium"
    out["post_rescue_rule"] = verdict_3way(c_new, verdict_cfg or Verdict3WayConfig())[1]
    out["reasons"] = list(out.get("reasons") or []) + ["No phishing evidence across all layers; hard legitimacy override applied."]
    return out


def build_dashboard_analysis(
    url: str,
    *,
    reinforcement: bool = True,
    layer1_use_dns: bool = False,
    ai_adjudication: bool = True,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Core analysis dict + evidence gap strings (no file write)."""
    ensure_layout()
    cfg = PipelineConfig.from_env()
    url = (url or "").strip()
    evidence_gaps: List[str] = []
    ml = predict_layer1(url, use_dns=layer1_use_dns)
    ml = _apply_official_brand_apex_cap(ml, url)
    if ml.get("error"):
        evidence_gaps.append("Layer-1 ML did not produce a score (missing model or feature error).")

    reinforcement_block: Optional[Dict[str, Any]] = None
    org_risk_raw = 0.0
    capture_json: Optional[Dict[str, Any]] = None
    click_probe_cfg_enabled = False
    if reinforcement:
        click_probe_cfg_enabled = bool(getattr(cfg, "enable_click_probe", False))
        try:
            cap = capture_url(url, cfg, namespace="suspicious")
            cj = cap.as_json()
            capture_json = cj
            org = org_style_from_capture_blob(cj, url)
            org_risk_raw = float(org.get("org_style_risk_score") or 0.0)
            reinforcement_block = {
                "capture": {
                    "error": cap.error,
                    "capture_blocked": cap.capture_blocked,
                    "capture_strategy": cap.capture_strategy,
                    "capture_block_reason": cap.capture_block_reason,
                    "final_url": cap.final_url,
                "title": cap.title,
                    "redirect_count": cap.redirect_count,
                    "cross_domain_redirect_count": cap.cross_domain_redirect_count,
                    "settled_successfully": cap.settled_successfully,
                    "html_path": cap.html_path,
                "visible_text_sample": (cap.visible_text or "")[:1200],
                },
                "org_style": org,
            }
            if cap.error or cap.capture_blocked or cap.capture_strategy in {"failed", "http_fallback"}:
                evidence_gaps.append(
                    "Live fetch/automation was limited; HTML/DOM-based reinforcement may be incomplete."
                )
        except Exception as e:  # noqa: BLE001
            logger.exception("reinforcement failed")
            reinforcement_block = {"error": str(e)}
            evidence_gaps.append(f"Reinforcement capture failed: {type(e).__name__}: {e}")
    else:
        evidence_gaps.append("Reinforcement skipped; verdict uses Layer-1 URL/host signals only.")

    snap = ml.get("brand_structure_features") or {}
    final_u = ""
    if capture_json:
        final_u = str(capture_json.get("final_url") or "")
    bundle = build_legitimacy_bundle(snap, final_url=final_u, input_url=url, capture_json=capture_json)

    html_path = (capture_json or {}).get("html_path") if capture_json else None
    soup = None
    if html_path:
        pth = Path(str(html_path))
        if pth.is_file():
            try:
                soup = BeautifulSoup(
                    pth.read_text(encoding="utf-8", errors="ignore"),
                    "html.parser",
                )
            except OSError:
                soup = None

    title_hint = str((capture_json or {}).get("title") or "")
    visible_hint = str((capture_json or {}).get("visible_text") or "")

    html_structure = extract_html_structure_signals(
        html_path=html_path,
        final_url=final_u,
        input_url=url,
        title_hint=title_hint,
        visible_text_hint=visible_hint,
        soup=soup,
    )
    html_dom = extract_html_dom_anomaly_signals(
        html_path=html_path,
        final_url=final_u,
        input_url=url,
        title_hint=title_hint,
        visible_text_hint=visible_hint,
        soup=soup,
    )
    layer2_enrichment, layer3_enrichment, strong_enrichment_signals, weak_enrichment_signals = _enrich_capture_and_html_signals(
        input_url=url,
        capture_json=capture_json,
        soup=soup,
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
    )
    if reinforcement_block and isinstance(reinforcement_block.get("capture"), dict):
        reinforcement_block["capture"] = {**reinforcement_block["capture"], **layer2_enrichment}
        _merge_capture_failure_fields(reinforcement_block["capture"], ml)
        _finalize_click_probe_diagnostics_on_capture(
            reinforcement_block["capture"],
            enable_click_probe=click_probe_cfg_enabled,
        )
        capd = reinforcement_block["capture"]
        if capd.get("capture_failure_suspicious"):
            gap_html = "HTML/DOM analysis was unavailable because live capture did not complete."
            if gap_html not in evidence_gaps:
                evidence_gaps.append(gap_html)
    if reinforcement_block and isinstance(reinforcement_block.get("org_style"), dict):
        damped_org = dampen_org_style_for_page_family(
            reinforcement_block["org_style"],
            html_dom.get("html_dom_anomaly_summary"),
        )
        reinforcement_block["org_style"] = damped_org
        org_risk_raw = float(damped_org.get("org_style_risk_score") or org_risk_raw)

    org_adj, org_extra_reasons = adjust_org_risk_for_legitimacy(org_risk_raw, bundle, [])
    if org_extra_reasons and reinforcement_block and isinstance(reinforcement_block.get("org_style"), dict):
        o = dict(reinforcement_block["org_style"])
        o["reasons"] = list(o.get("reasons") or []) + org_extra_reasons
        reinforcement_block["org_style"] = o

    p_raw = ml.get("phish_proba_model_raw")
    if p_raw is None and ml.get("phish_proba") is not None:
        p_raw = ml.get("phish_proba")
    p_cal = ml.get("phish_proba_calibrated", p_raw)

    base_ml = float(ml["phish_proba"]) if ml.get("phish_proba") is not None else None
    ml_eff, blend_meta = (
        blend_ml_phish_for_legitimacy(base_ml, org_adj, bundle) if base_ml is not None else (None, {})
    )
    host_path_payload = assess_host_path_reasoning(
        input_url=url,
        final_url=final_u,
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        legitimacy_bundle=bundle,
    )
    host_path_reasoning = host_path_payload.get("host_path_reasoning")
    host_path_error = host_path_payload.get("host_path_reasoning_error")
    ml_eff, host_path_blend_meta = blend_ml_phish_for_host_path_reasoning(
        phish_proba=ml_eff,
        host_path_reasoning=host_path_reasoning,
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        legitimacy_bundle=bundle,
    )
    blend_meta = {**blend_meta, **host_path_blend_meta}
    ml_eff, dns_dampening_meta = _apply_dns_feature_dominance_dampening(
        ml_effective_score=ml_eff,
        ml=ml,
        layer2_capture=(reinforcement_block or {}).get("capture"),
        html_structure_risk=html_structure.get("html_structure_risk_score"),
        html_dom_risk=html_dom.get("html_dom_anomaly_risk_score"),
        legitimacy_bundle=bundle,
        cfg=cfg,
    )
    capture_for_policy = dict(((reinforcement_block or {}).get("capture") or {}))
    platform_context = _classify_platform_host_context(
        layer2_capture=capture_for_policy,
        html_dom_enrichment=layer3_enrichment,
        host_path_reasoning=host_path_reasoning,
        cfg=cfg,
    )
    pctx_type = str(platform_context.get("platform_context_type") or "")
    oauth_detected = bool(platform_context.get("oauth_providers_detected"))
    prelim_blockers = _compute_phishing_blockers(
        host_path_reasoning=host_path_reasoning,
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        layer2_capture=capture_for_policy,
        legitimacy_bundle=bundle,
        cloud_hosted_brand_impersonation=bool(pctx_type == "cloud_hosted_brand_impersonation"),
        suppress_oauth_brand_mismatch=False,
        platform_context_type=pctx_type,
    )
    non_brand_blockers = [b for b in prelim_blockers if b != "brand_domain_mismatch"]
    suppress_oauth_brand_mismatch = bool(
        pctx_type in {"official_platform_domain", "official_platform_login"}
        and oauth_detected
        and bool(capture_for_policy.get("brand_domain_mismatch"))
        and not non_brand_blockers
    )
    if suppress_oauth_brand_mismatch:
        capture_for_policy["brand_domain_mismatch"] = False
        platform_context["oauth_brand_mismatch_suppressed"] = True
        platform_context["platform_context_reasons"] = list(platform_context.get("platform_context_reasons") or []) + [
            "OAuth-provider references on official platform context suppressed brand mismatch only after blocker checks."
        ]
    if reinforcement_block is not None:
        reinforcement_block["capture"] = capture_for_policy
    trust_blockers = _compute_phishing_blockers(
        host_path_reasoning=host_path_reasoning,
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        layer2_capture=capture_for_policy,
        legitimacy_bundle=bundle,
        cloud_hosted_brand_impersonation=bool(
            str(platform_context.get("platform_context_type") or "") == "cloud_hosted_brand_impersonation"
        ),
        suppress_oauth_brand_mismatch=suppress_oauth_brand_mismatch,
        platform_context_type=str(platform_context.get("platform_context_type") or "unknown"),
    )
    hosting_trust = _evaluate_hosting_domain_trust(
        layer2_capture=capture_for_policy,
        html_structure_risk=html_structure.get("html_structure_risk_score"),
        html_dom_risk=html_dom.get("html_dom_anomaly_risk_score"),
        blockers=trust_blockers,
        cfg=cfg,
    )
    ml_eff, ml_overcap_meta = _apply_ml_overconfidence_cap(
        ml_effective_score=ml_eff,
        layer2_capture=capture_for_policy,
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        html_structure_risk=html_structure.get("html_structure_risk_score"),
        html_dom_risk=html_dom.get("html_dom_anomaly_risk_score"),
        host_path_reasoning=host_path_reasoning,
        platform_context=platform_context,
        hosting_trust=hosting_trust,
    )
    blend_meta = {**blend_meta, "dns_feature_dampening": dns_dampening_meta, "ml_overconfidence_cap": ml_overcap_meta}

    verdict = _verdict_from_scores(
        ml_eff,
        float(p_raw) if p_raw is not None else None,
        float(p_cal) if p_cal is not None else None,
        org_risk_raw,
        org_adj,
        verdict_cfg=verdict_cfg,
    )
    if bool(ml_overcap_meta.get("ml_overconfidence_cap_applied")):
        verdict["reasons"] = list(verdict.get("reasons") or []) + [
            str(ml_overcap_meta.get("ml_overconfidence_cap_reason") or "")
        ]

    verdict = _apply_legitimacy_rescue_on_verdict(
        verdict,
        ml=ml,
        host_path_reasoning=host_path_reasoning,
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_structure_risk=html_structure.get("html_structure_risk_score"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        html_dom_risk=html_dom.get("html_dom_anomaly_risk_score"),
        layer2_capture=capture_for_policy,
        legitimacy_bundle=bundle,
        cfg=cfg,
        verdict_cfg=verdict_cfg,
    )
    if "dormant_phishing_infra" in list(platform_context.get("platform_context_blockers") or []):
        trust_blockers = list(trust_blockers) + ["dormant_phishing_infra"]
        platform_context["dormant_phishing_infra_detected"] = True
        platform_context["dormant_phishing_infra_reasons"] = [
            "Inactive page on suspicious user-hosted infrastructure consistent with dormant phishing setup."
        ]
    verdict = _apply_hosting_trust_promotion(
        verdict,
        trust=hosting_trust,
        ml=ml,
        verdict_cfg=verdict_cfg,
    )
    verdict = _apply_platform_context_policy(
        verdict,
        platform_context=platform_context,
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        html_structure_risk=html_structure.get("html_structure_risk_score"),
        html_dom_risk=html_dom.get("html_dom_anomaly_risk_score"),
        verdict_cfg=verdict_cfg,
    )
    verdict = _apply_untrusted_builder_hosting_downgrade(
        verdict,
        layer2_capture=capture_for_policy,
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        html_dom_enrichment=layer3_enrichment,
        blockers=trust_blockers,
    )
    cap_for_guard = (reinforcement_block or {}).get("capture") or {}
    capture_failed_g = bool(cap_for_guard.get("capture_failed"))
    hse_raw = html_structure.get("html_structure_error")
    hse_s = hse_raw if isinstance(hse_raw, str) else None
    hde_raw = html_dom.get("html_dom_anomaly_error")
    hde_s = hde_raw if isinstance(hde_raw, str) else None
    hcmr_raw = layer3_enrichment.get("html_capture_missing_reason")
    hcmr_s = hcmr_raw if isinstance(hcmr_raw, str) else None
    html_missing_for_reinforcement = _html_missing_after_capture_failure(
        capture_failed=capture_failed_g,
        html_capture_missing_reason=hcmr_s,
        html_structure_error=hse_s,
        html_dom_anomaly_error=hde_s,
    )
    verdict = _apply_capture_failure_verdict_hardening(
        verdict,
        ml=ml,
        legitimacy_bundle=bundle,
        capture_failed=capture_failed_g,
        html_missing_for_reinforcement=html_missing_for_reinforcement,
        verdict_cfg=verdict_cfg,
    )
    if isinstance(verdict.get("combined_score"), (int, float)):
        vlabel, vwhy = verdict_3way(float(verdict["combined_score"]), verdict_cfg or Verdict3WayConfig())
        verdict["label"] = vlabel
        verdict["verdict_3way"] = vlabel
        verdict["confidence"] = "medium" if vlabel != "uncertain" else "low"
        verdict["post_rescue_rule"] = vwhy
    verdict["effective_ml_score"] = ml_eff
    verdict["legitimacy_bundle"] = bundle
    verdict["ml_legitimacy_blend"] = blend_meta
    oc = (blend_meta or {}).get("ml_overconfidence_cap") or {}
    verdict["ml_overconfidence_cap_applied"] = bool(oc.get("ml_overconfidence_cap_applied"))
    verdict["ml_overconfidence_cap_reason"] = oc.get("ml_overconfidence_cap_reason")
    verdict["ml_score_before_overconfidence_cap"] = oc.get("ml_score_before_overconfidence_cap")
    verdict["ml_score_after_overconfidence_cap"] = oc.get("ml_score_after_overconfidence_cap")
    verdict["html_structure_risk_score"] = html_structure.get("html_structure_risk_score")
    verdict["html_structure_reasons"] = html_structure.get("html_structure_reasons")
    verdict["html_dom_anomaly_risk_score"] = html_dom.get("html_dom_anomaly_risk_score")
    verdict["html_dom_anomaly_reasons"] = html_dom.get("html_dom_anomaly_reasons")
    verdict["html_dom_visual_assessment"] = html_dom.get("html_dom_visual_assessment")
    verdict["host_path_identity_class"] = (host_path_reasoning or {}).get("host_identity_class")
    verdict["host_path_legitimacy_confidence"] = (host_path_reasoning or {}).get("host_legitimacy_confidence")
    verdict["host_path_fit_assessment"] = (host_path_reasoning or {}).get("path_fit_assessment")
    try:
        ml_cal_guard = float(p_cal) if p_cal is not None else None
    except (TypeError, ValueError):
        ml_cal_guard = None
    guard_on = no_phishing_evidence_guard(
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        html_dom_risk=html_dom.get("html_dom_anomaly_risk_score"),
        host_path_reasoning=host_path_reasoning,
        capture_failed=capture_failed_g,
        html_structure_error=hse_s,
        html_capture_missing_reason=hcmr_s,
        ml_calibrated_phish=ml_cal_guard,
    )
    verdict = _apply_no_phishing_evidence_override(verdict, guard_triggered=guard_on, verdict_cfg=verdict_cfg)

    pre_ml_capture_miss_safety_verdict = str(verdict.get("verdict_3way") or verdict.get("label") or "")
    verdict = _apply_ml_phishing_capture_miss_legitimacy_safety(
        verdict,
        ml=ml,
        legitimacy_bundle=bundle,
        capture_failed=capture_failed_g,
        html_capture_missing_reason=hcmr_s,
        html_structure_error=hse_s,
        html_dom_anomaly_error=hde_s,
        verdict_cfg=verdict_cfg,
    )

    ai_block = run_ai_adjudication(
        input_url=url,
        layer1_ml=ml,
        reinforcement=reinforcement_block,
        verdict_pre_ai=verdict,
        legitimacy_bundle=bundle,
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_structure_risk_score=html_structure.get("html_structure_risk_score"),
        html_structure_reasons=html_structure.get("html_structure_reasons"),
        html_dom_anomaly_summary=html_dom.get("html_dom_anomaly_summary"),
        html_dom_anomaly_risk_score=html_dom.get("html_dom_anomaly_risk_score"),
        html_dom_anomaly_reasons=html_dom.get("html_dom_anomaly_reasons"),
        html_dom_visual_assessment=html_dom.get("html_dom_visual_assessment"),
        host_path_reasoning=host_path_reasoning,
        evidence_gaps=evidence_gaps,
        capture_json=capture_json,
        enabled=ai_adjudication,
        verdict_cfg=verdict_cfg,
        html_capture_missing_reason=hcmr_s,
        html_structure_error=hse_s,
        html_dom_anomaly_error=hde_s,
        pre_verdict_before_ml_capture_miss_safety=pre_ml_capture_miss_safety_verdict,
    )
    verdict["ai_adjudication"] = ai_block
    if ai_block.get("ran") and isinstance(ai_block.get("adjustment"), dict):
        adj = ai_block["adjustment"]
        if adj.get("applied"):
            post_score = adj.get("post_ai_score")
            post_verdict = adj.get("post_ai_verdict")
            if isinstance(post_score, (int, float)):
                verdict["combined_score_pre_ai"] = verdict.get("combined_score")
                verdict["combined_score"] = float(post_score)
                verdict["ai_adjustment_applied"] = float(adj.get("adjustment_applied") or 0.0)
            if post_verdict:
                verdict["verdict_3way_pre_ai"] = verdict.get("verdict_3way")
                verdict["verdict_3way"] = post_verdict
                verdict["label"] = post_verdict
                verdict["confidence"] = "medium" if post_verdict != "uncertain" else "low"
                verdict["reasons"] = list(verdict.get("reasons") or []) + [
                    f"AI adjudication applied bounded adjustment {float(adj.get('adjustment_applied') or 0.0):+.3f}; post-AI verdict={post_verdict}."
                ]
    # Hard no-phishing-evidence override always applies last.
    verdict = _apply_no_phishing_evidence_override(verdict, guard_triggered=guard_on, verdict_cfg=verdict_cfg)
    verdict = _apply_ml_phishing_capture_miss_legitimacy_safety(
        verdict,
        ml=ml,
        legitimacy_bundle=bundle,
        capture_failed=capture_failed_g,
        html_capture_missing_reason=hcmr_s,
        html_structure_error=hse_s,
        html_dom_anomaly_error=hde_s,
        verdict_cfg=verdict_cfg,
    )
    verdict = _apply_inactive_site_overlay(
        verdict,
        ml=ml,
        layer2_capture=capture_for_policy,
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        html_dom_enrichment=layer3_enrichment,
        blockers=trust_blockers,
    )

    out: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "input_url": url,
        "layer1_ml": ml,
        "reinforcement": reinforcement_block,
        "html_structure": html_structure,
        "html_dom_anomaly": html_dom,
        "html_dom_enrichment": layer3_enrichment,
        "enrichment_signals": {
            "strong_signals": strong_enrichment_signals,
            "weak_contextual_signals": weak_enrichment_signals,
        },
        "host_path_reasoning": host_path_reasoning,
        "host_path_reasoning_error": host_path_error,
        "verdict": verdict,
        "evidence_gaps": evidence_gaps,
    }
    return out, evidence_gaps


def analyze_url_dashboard(
    url: str,
    *,
    reinforcement: bool = True,
    layer1_use_dns: bool = False,
    ai_adjudication: bool = True,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    out, _ = build_dashboard_analysis(
        url,
        reinforcement=reinforcement,
        layer1_use_dns=layer1_use_dns,
        ai_adjudication=ai_adjudication,
        verdict_cfg=verdict_cfg,
    )
    analysis_dir().mkdir(parents=True, exist_ok=True)
    (analysis_dir() / "last_dashboard_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Dashboard analysis JSON (ML + optional reinforcement).")
    ap.add_argument("--url", required=True)
    ap.add_argument("--no-reinforcement", action="store_true")
    ap.add_argument("--layer1-use-dns", action="store_true")
    ap.add_argument("--no-ai-adjudication", action="store_true")
    args = ap.parse_args()
    row = analyze_url_dashboard(
        args.url,
        reinforcement=not args.no_reinforcement,
        layer1_use_dns=args.layer1_use_dns,
        ai_adjudication=not args.no_ai_adjudication,
    )
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
