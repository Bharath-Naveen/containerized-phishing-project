"""Lightweight JavaScript and network behavior risk signals for EAL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

import tldextract
from bs4 import BeautifulSoup

from .domain_ecosystem import domain_relation

_BENIGN_SUFFIXES = (
    "googleapis.com",
    "gstatic.com",
    "google-analytics.com",
    "googletagmanager.com",
    "cloudflare.com",
    "cloudfront.net",
    "jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "jquery.com",
    "stripe.com",
)

_JS_LOGIN_TERMS = ("password", "email", "login", "user", "pass", "auth", "account", "credential")
_CRED_PATH_TERMS = ("login", "auth", "token", "verify", "password", "credential", "session", "account", "submit", "collect")


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _registrable(host: str) -> str:
    ext = tldextract.extract((host or "").lower())
    return ".".join(p for p in (ext.domain, ext.suffix) if p).lower()


def _is_benign_domain(reg: str, *, final_reg: str, payment_context_legit: bool) -> bool:
    if not reg:
        return False
    if reg == "paypal.com":
        return bool(payment_context_legit or final_reg.endswith("paypal.com"))
    return any(reg.endswith(sfx) for sfx in _BENIGN_SUFFIXES)


def _is_unrelated_domain(*, final_host: str, target_host: str) -> bool:
    rel = domain_relation(final_host, target_host)
    return bool(rel.get("truly_external"))


def _extract_script_texts(html: str) -> List[str]:
    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    out: List[str] = []
    for s in soup.find_all("script"):
        txt = str(s.string or s.get_text(" ", strip=False) or "")
        if txt:
            out.append(txt)
    return out


def _extract_js_domains(pattern: str, script_blob: str) -> Set[str]:
    out: Set[str] = set()
    for m in re.finditer(pattern, script_blob, flags=re.IGNORECASE):
        u = str(m.group(1) or "")
        h = _host(u)
        if h:
            out.add(h)
    return out


def extract_behavior_signals(
    *,
    html_path: str | None,
    layer2_capture: Dict[str, Any] | None,
    html_structure_summary: Dict[str, Any] | None,
    html_dom_summary: Dict[str, Any] | None,
    platform_context_type: str = "unknown",
) -> Dict[str, Any]:
    cap = layer2_capture or {}
    hs = html_structure_summary or {}
    dom = html_dom_summary or {}
    final_url = str(cap.get("final_url") or "")
    final_host = _host(final_url)
    final_reg = str(cap.get("final_registered_domain") or _registrable(final_host)).lower()

    interaction = cap.get("interaction") or {}
    submit_or_probe = bool(interaction.get("attempted_submit") or interaction.get("click_probe_attempted"))
    form_ctx = bool(int(hs.get("password_input_count") or 0) > 0 or int(hs.get("form_count") or 0) > 0)
    cred_ctx = bool(form_ctx or int(dom.get("form_action_external_domain_count") or 0) > 0)
    payment_context_legit = bool(platform_context_type in {"official_platform_domain", "official_platform_login"} and not cap.get("brand_domain_mismatch"))

    html_text = ""
    if html_path:
        p = Path(str(html_path))
        if p.is_file():
            try:
                html_text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                html_text = ""
    scripts = _extract_script_texts(html_text)
    script_blob = "\n".join(scripts)

    signals: List[str] = []
    obf_count = 0
    if re.search(r"eval\s*\(\s*atob\s*\(", script_blob, flags=re.IGNORECASE):
        obf_count += 2
        signals.append("eval_atob_pattern")
    if re.search(r"eval\s*\(\s*unescape\s*\(", script_blob, flags=re.IGNORECASE):
        obf_count += 2
        signals.append("eval_unescape_pattern")
    if re.search(r"String\.fromCharCode\s*\((?:[^)]*,){7,}[^)]*\)", script_blob, flags=re.IGNORECASE):
        obf_count += 2
        signals.append("fromcharcode_chain")
    if re.search(r"(?:[A-Za-z0-9+/]{140,}={0,2}|(?:0x)?[A-Fa-f0-9]{180,})", script_blob):
        obf_count += 1
        signals.append("long_encoded_string")

    anti_debug = bool(
        re.search(r"set(?:Interval|Timeout)\s*\([\s\S]{0,240}?debugger", script_blob, flags=re.IGNORECASE)
    )
    if anti_debug:
        signals.append("anti_debugger_timer")

    dyn_form = bool(
        re.search(r"createElement\s*\(\s*['\"]input['\"]\s*\)", script_blob, flags=re.IGNORECASE)
        and any(t in script_blob.lower() for t in _JS_LOGIN_TERMS)
    )
    if dyn_form:
        signals.append("dynamic_input_injection")

    fetch_hosts = _extract_js_domains(r"(?:fetch|open)\s*\([^)]*['\"](https?://[^'\"]+)", script_blob)
    redir_hosts = _extract_js_domains(r"window\.location(?:\.href)?\s*=\s*['\"](https?://[^'\"]+)", script_blob)

    suspicious_fetch_regs: Set[str] = set()
    for h in fetch_hosts:
        reg = _registrable(h)
        if not reg:
            continue
        if _is_benign_domain(reg, final_reg=final_reg, payment_context_legit=payment_context_legit):
            continue
        if _is_unrelated_domain(final_host=final_host, target_host=h):
            suspicious_fetch_regs.add(reg)

    suspicious_redirect_regs: Set[str] = set()
    for h in redir_hosts:
        reg = _registrable(h)
        if not reg:
            continue
        if _is_unrelated_domain(final_host=final_host, target_host=h):
            suspicious_redirect_regs.add(reg)

    if suspicious_fetch_regs:
        signals.append("suspicious_js_fetch_unrelated_domain")
    if suspicious_redirect_regs:
        signals.append("suspicious_js_redirect_unrelated_domain")

    req_urls = cap.get("network_request_urls") or []
    req_hosts: Set[str] = set()
    for u in req_urls:
        h = _host(str(u))
        if h:
            req_hosts.add(h)

    benign_domains: Set[str] = set()
    unrelated_domains: Set[str] = set()
    third_party_domains: Set[str] = set()
    for h in req_hosts:
        reg = _registrable(h)
        if not reg:
            continue
        if reg != final_reg:
            third_party_domains.add(reg)
        if _is_benign_domain(reg, final_reg=final_reg, payment_context_legit=payment_context_legit):
            benign_domains.add(reg)
            continue
        if _is_unrelated_domain(final_host=final_host, target_host=h):
            unrelated_domains.add(reg)

    exfil_reasons: List[str] = []
    exfil = False
    if cred_ctx and submit_or_probe:
        for u in req_urls:
            su = str(u).lower()
            h = _host(su)
            if not h:
                continue
            reg = _registrable(h)
            if not reg or reg in benign_domains:
                continue
            if _is_unrelated_domain(final_host=final_host, target_host=h) and any(t in su for t in _CRED_PATH_TERMS):
                exfil = True
                exfil_reasons.append(f"credential_like_request_to_unrelated_domain:{reg}")
                break

    score = min(1.0, round(obf_count / 6.0, 4))
    unavailable = bool((not html_text) and not req_urls)

    return {
        "js_obfuscation_score": float(score),
        "js_obfuscation_signals": signals,
        "js_dynamic_form_injection_detected": bool(dyn_form),
        "js_anti_debugging_detected": bool(anti_debug),
        "js_suspicious_fetch_domains": sorted(suspicious_fetch_regs),
        "js_suspicious_redirect_domains": sorted(suspicious_redirect_regs),
        "network_request_domain_count": int(len(req_hosts)),
        "network_third_party_domains": sorted(third_party_domains),
        "network_unrelated_domains": sorted(unrelated_domains),
        "network_common_benign_domains": sorted(benign_domains),
        "network_exfiltration_suspected": bool(exfil),
        "network_exfiltration_reasons": exfil_reasons,
        "behavior_analysis_unavailable": unavailable,
    }
