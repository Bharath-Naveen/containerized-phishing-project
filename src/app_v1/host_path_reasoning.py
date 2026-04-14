"""Host identity + path conformity reasoning for public-host legitimacy checks.

This layer is structured and bounded: it does not force verdicts, but provides
auditable signals and optional mild ML score discounts when host legitimacy is
high and path behavior is plausible for that host.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlparse

import tldextract

from src.pipeline.features.brand_signals import BRAND_TOKENS, host_on_official_brand_apex
from src.pipeline.features.hosting_features import extract_hosting_features
from src.pipeline.safe_url import safe_urlparse

_TRUST_ACTION_TERMS = (
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "account",
    "recover",
    "reset",
    "password",
    "billing",
    "payment",
    "wallet",
    "confirm",
    "secure",
)

_CONTENT_PATH_HINTS = (
    "/r/",
    "/u/",
    "/user/",
    "/users/",
    "/posts/",
    "/post/",
    "/article/",
    "/news/",
    "/story/",
    "/feed",
    "/thread/",
    "/topic/",
    "/community/",
    "/forum/",
    "/tag/",
    "/tags/",
    "/questions/",
)

_DOCS_HINTS = ("/docs", "/doc/", "/wiki/", "/kb/", "/knowledge", "/reference/", "/manual/", "/api/")
_TRUST_INTENT_PATH_RE = re.compile(
    r"/(login|signin|sign-in|verify|verification|recover|reset|password|billing|payment|wallet|checkout)(/|$|[-_])",
    re.I,
)


def _registrable(hostname: str) -> str:
    ext = tldextract.extract((hostname or "").lower())
    return ".".join(p for p in (ext.domain, ext.suffix) if p).lower()


def _subdomain_labels(hostname: str, registrable: str) -> List[str]:
    h = (hostname or "").lower().strip(".")
    r = (registrable or "").lower().strip(".")
    if not h or not r:
        return []
    if h == r:
        return []
    suffix = "." + r
    left = h[: -len(suffix)] if h.endswith(suffix) else h
    return [x for x in left.split(".") if x]


def _query_summary(query: str) -> Dict[str, Any]:
    pairs = parse_qsl(query or "", keep_blank_values=True)
    redirect_like = 0
    long_values = 0
    for k, v in pairs:
        lv = (v or "").lower()
        lk = (k or "").lower()
        if len(v or "") >= 80:
            long_values += 1
        if lk in {"next", "redirect", "return", "return_to", "continue", "url", "target", "dest", "destination"}:
            redirect_like += 1
        elif "http://" in lv or "https://" in lv:
            redirect_like += 1
    return {
        "param_count": len(pairs),
        "redirect_like_param_count": redirect_like,
        "long_value_param_count": long_values,
        "param_keys_sample": [k for k, _ in pairs][:8],
    }


def _host_suspicious_reasons(parsed: Any, host: str, registrable: str, sublabels: List[str]) -> List[str]:
    reasons: List[str] = []
    netloc = (parsed.netloc or "").lower() if parsed else ""
    if "@" in netloc:
        reasons.append("userinfo_in_netloc")
    if host.startswith("xn--") or any(x.startswith("xn--") for x in sublabels):
        reasons.append("punycode_label_present")
    if re.search(r"\d{5,}", host):
        reasons.append("numeric_heavy_hostname")
    if any("--" in x for x in sublabels):
        reasons.append("double_hyphen_subdomain")
    if host.count(".") >= 4:
        reasons.append("very_deep_subdomain_chain")
    if "%" in netloc:
        reasons.append("encoded_netloc_sequence")
    try:
        ipaddress.ip_address(host)
        reasons.append("ip_literal_host")
    except ValueError:
        pass

    reg = (registrable or host).lower()
    for tok in BRAND_TOKENS:
        if tok in reg:
            continue
        if any(tok in s for s in sublabels):
            reasons.append("brand_token_in_subdomain_not_matching_registrable")
            break
    return reasons


def assess_host_path_reasoning(
    *,
    input_url: str,
    final_url: str = "",
    html_dom_summary: Optional[Dict[str, Any]] = None,
    legitimacy_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured host identity and path-fit assessment."""
    target_url = (final_url or input_url or "").strip()
    parsed, parse_err = safe_urlparse(target_url)
    if parse_err or parsed is None:
        return {
            "host_path_reasoning": None,
            "host_path_reasoning_error": f"url_parse_error:{parse_err}",
        }

    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower()
    registrable = _registrable(hostname)
    sublabels = _subdomain_labels(hostname, registrable)
    path = parsed.path or "/"
    path_l = path.lower()
    qsum = _query_summary(parsed.query or "")

    dom = html_dom_summary or {}
    page_family = str(dom.get("page_family") or "")
    trust_action_context = bool(dom.get("trust_action_context"))
    has_form_external = int(dom.get("form_action_external_domain_count") or 0) > 0
    login_harvester = bool(dom.get("login_harvester_pattern"))
    wrapper_pattern = bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern"))
    strong_anchor_mismatch = int(dom.get("anchor_strong_mismatch_count") or 0) > 0

    host_reasons = _host_suspicious_reasons(parsed, hostname, registrable, sublabels)
    hosting = extract_hosting_features(target_url)
    free_hosting_flag = bool(int(hosting.get("free_hosting_flag") or 0))

    identity_class = "generic_unknown_host"
    confidence = "medium"
    legitimacy_reasons: List[str] = []

    if host_reasons:
        identity_class = "suspicious_host_pattern"
        confidence = "low"
        legitimacy_reasons.append("Suspicious hostname/netloc pattern indicators present.")
    elif host_on_official_brand_apex(hostname):
        identity_class = "official_brand_auth"
        confidence = "high"
        legitimacy_reasons.append("Host is under a known official registrable brand apex.")
    elif page_family in {"content_feed_forum_aggregator", "article_news"}:
        identity_class = "social_forum_feed" if page_family == "content_feed_forum_aggregator" else "public_content_platform"
        confidence = "high" if not free_hosting_flag else "medium"
        legitimacy_reasons.append("DOM/page-family resembles a public content or discussion surface.")
    elif any(h in path_l for h in _DOCS_HINTS):
        identity_class = "public_docs_or_reference"
        confidence = "high" if not free_hosting_flag else "medium"
        legitimacy_reasons.append("Path resembles documentation/reference content.")
    elif page_family == "dashboard_admin":
        identity_class = "dashboard_admin"
        confidence = "medium"
        legitimacy_reasons.append("Path/page-family resembles dashboard or admin surface.")
    else:
        if registrable and "." in registrable and not free_hosting_flag:
            confidence = "medium"
            legitimacy_reasons.append("Host structure appears well-formed with registrable public domain.")
        else:
            confidence = "low"
            legitimacy_reasons.append("Host lacks clear high-confidence public legitimacy anchors.")

    path_fit = "unusual_but_possible"
    path_reasons: List[str] = []
    trust_path = any(t in path_l for t in _TRUST_ACTION_TERMS)
    trust_path_intent = bool(_TRUST_INTENT_PATH_RE.search(path_l))
    content_hint = any(h in path_l for h in _CONTENT_PATH_HINTS)
    docs_hint = any(h in path_l for h in _DOCS_HINTS)
    brand_in_path = any(b in path_l for b in BRAND_TOKENS)

    if identity_class == "suspicious_host_pattern":
        path_fit = "suspicious"
        path_reasons.append("Host pattern itself is suspicious regardless of path semantics.")
    elif wrapper_pattern:
        path_fit = "suspicious"
        path_reasons.append("Wrapper/interstitial behavior is inconsistent with ordinary content navigation.")
    elif has_form_external or login_harvester or strong_anchor_mismatch:
        path_fit = "suspicious"
        path_reasons.append("Trust-action capture mismatch exists (external form/action mismatch or harvester signal).")
    elif (
        identity_class in {"public_content_platform", "social_forum_feed", "public_docs_or_reference"}
        and trust_path_intent
        and not trust_action_context
    ):
        path_fit = "suspicious"
        path_reasons.append(
            "Trust-action intent path appears on content/reference host without matching credential/trust surface."
        )
    elif brand_in_path and not host_on_official_brand_apex(hostname) and trust_path and trust_action_context:
        path_fit = "suspicious"
        path_reasons.append("Path mixes brand/trust terms that do not align with host identity and trust-action context.")
    elif qsum["redirect_like_param_count"] >= 2 and trust_path:
        path_fit = "suspicious"
        path_reasons.append("Query contains multiple redirect/target style parameters in trust-action path.")
    elif identity_class in {"public_content_platform", "social_forum_feed"} and content_hint:
        path_fit = "plausible"
        path_reasons.append("Path matches common public content/forum navigation style.")
    elif identity_class == "public_docs_or_reference" and docs_hint:
        path_fit = "plausible"
        path_reasons.append("Path aligns with docs/reference style navigation.")
    elif identity_class == "official_brand_auth" and (trust_path or path_l in {"/", ""}):
        path_fit = "plausible"
        path_reasons.append("Path is plausible for official host identity.")
    elif path_l in {"", "/"}:
        path_fit = "plausible"
        path_reasons.append("Root path is neutral/plausible for public host entry point.")
    else:
        path_reasons.append("Path does not strongly confirm legitimacy but is not directly suspicious.")

    return {
        "host_path_reasoning": {
            "url_decomposition": {
                "scheme": scheme,
                "hostname": hostname,
                "registrable_domain": registrable,
                "subdomain_labels": sublabels,
                "normalized_path": path if path else "/",
                "query_summary": qsum,
            },
            "host_identity_class": identity_class,
            "host_legitimacy_confidence": confidence,
            "host_legitimacy_reasons": legitimacy_reasons + host_reasons,
            "path_fit_assessment": path_fit,
            "path_fit_reasons": path_reasons,
        },
        "host_path_reasoning_error": None,
    }


def blend_ml_phish_for_host_path_reasoning(
    *,
    phish_proba: Optional[float],
    host_path_reasoning: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    legitimacy_bundle: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Bounded ML-score reduction for high-confidence legitimate host+path context."""
    meta: Dict[str, Any] = {
        "host_path_ml_discount_applied": False,
        "host_path_ml_discount": 0.0,
        "host_path_discount_reason": None,
        "host_path_legit_rescue_tier": None,
    }
    if phish_proba is None:
        return None, meta

    p = float(phish_proba)
    if p >= 0.85:
        return p, meta

    hp = host_path_reasoning or {}
    conf = str(hp.get("host_legitimacy_confidence") or "")
    pfit = str(hp.get("path_fit_assessment") or "")
    hclass = str(hp.get("host_identity_class") or "")

    dom = html_dom_summary or {}
    trust_action_context = bool(dom.get("trust_action_context"))
    suspicious_form = bool(int(dom.get("form_action_external_domain_count") or 0) > 0)
    login_harvester = bool(dom.get("login_harvester_pattern"))
    wrapper_pattern = bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern"))
    strong_anchor_mismatch = bool(int(dom.get("anchor_strong_mismatch_count") or 0) > 0)
    suspicious_cred = bool(dom.get("suspicious_credential_collection_pattern"))

    blocked = any(
        (
            hclass == "suspicious_host_pattern",
            pfit == "suspicious",
            trust_action_context,
            suspicious_form,
            login_harvester,
            wrapper_pattern,
            strong_anchor_mismatch,
            suspicious_cred,
            bool(legitimacy_bundle.get("suspicious_form_action_cross_origin")),
            not bool(legitimacy_bundle.get("no_deceptive_token_placement", True)),
            not bool(legitimacy_bundle.get("no_free_hosting_signal", True)),
        )
    )
    if blocked:
        return p, meta

    if conf != "high" or pfit != "plausible":
        return p, meta

    discount = 0.08
    tier = "base"
    if hclass in {"public_content_platform", "social_forum_feed", "public_docs_or_reference"}:
        discount = 0.14
        tier = "public_content"
    if hclass in {"public_content_platform", "social_forum_feed", "public_docs_or_reference"} and p >= 0.58:
        discount = 0.18
        tier = "high_ml_public_content"
    if p <= 0.30:
        discount = min(discount, 0.06)
    if p >= 0.78:
        discount = min(0.22, max(discount, 0.20))
        tier = "very_high_ml_public_content"

    p2 = max(0.0, p * (1.0 - discount))
    meta["host_path_ml_discount_applied"] = True
    meta["host_path_ml_discount"] = round(discount, 4)
    meta["host_path_legit_rescue_tier"] = tier
    meta["host_path_discount_reason"] = (
        "High-confidence host identity with plausible path and no trust-action/credential-capture indicators (bounded rescue discount)."
    )
    return p2, meta
