"""Loaders (with path-keyed caches) for the trusted-domain, platform-domain and official-domain registries, plus platform host classification.

Split out of the former analyze_dashboard.py in rebuild Phase 2 (code moved unchanged).
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from phishguard.paths import project_root
from .runtime_config import PipelineConfig
from .domain_utils import (
    _creator_platform_host_candidate,
    _detect_cloud_hosted_brand_impersonation,
    _hostname,
    _safe_subdomain_allowed,
)


_TRUSTED_DOMAIN_REGISTRY_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


_TRUSTED_DOMAIN_REGISTRY_CACHE_PATH: Optional[str] = None


_PLATFORM_DOMAIN_REGISTRY_CACHE: Optional[List[Dict[str, Any]]] = None


_PLATFORM_DOMAIN_REGISTRY_CACHE_PATH: Optional[str] = None


_OFFICIAL_DOMAIN_TRUST_PRIOR_CACHE: Optional[Dict[str, Any]] = None


_REPO_ROOT = project_root()


_OFFICIAL_DOMAINS_JSON_PATH = _REPO_ROOT / "data" / "official_domains.json"


def _load_trusted_domain_registry(csv_path: str) -> Dict[str, Dict[str, Any]]:
    # Cache is keyed by resolved path (same fix the platform registry got earlier): a cache that
    # ignored the path could be filled once from a different CSV and then served everywhere.
    global _TRUSTED_DOMAIN_REGISTRY_CACHE
    global _TRUSTED_DOMAIN_REGISTRY_CACHE_PATH
    p = Path(csv_path)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    resolved = str(p)
    if _TRUSTED_DOMAIN_REGISTRY_CACHE is not None and _TRUSTED_DOMAIN_REGISTRY_CACHE_PATH == resolved:
        return _TRUSTED_DOMAIN_REGISTRY_CACHE
    _TRUSTED_DOMAIN_REGISTRY_CACHE_PATH = resolved
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
    global _PLATFORM_DOMAIN_REGISTRY_CACHE_PATH
    p = Path(csv_path)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    resolved = str(p)
    if _PLATFORM_DOMAIN_REGISTRY_CACHE is not None and _PLATFORM_DOMAIN_REGISTRY_CACHE_PATH == resolved:
        return _PLATFORM_DOMAIN_REGISTRY_CACHE
    if not p.is_file():
        _PLATFORM_DOMAIN_REGISTRY_CACHE = []
        _PLATFORM_DOMAIN_REGISTRY_CACHE_PATH = resolved
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
    _PLATFORM_DOMAIN_REGISTRY_CACHE_PATH = resolved
    return rows


def _load_official_domain_trust_prior_registry() -> Dict[str, Any]:
    global _OFFICIAL_DOMAIN_TRUST_PRIOR_CACHE
    if _OFFICIAL_DOMAIN_TRUST_PRIOR_CACHE is not None:
        return _OFFICIAL_DOMAIN_TRUST_PRIOR_CACHE
    out: Dict[str, Any] = {"root_domains": set(), "canonical_subdomains": {}}
    p = _OFFICIAL_DOMAINS_JSON_PATH
    if not p.is_file():
        _OFFICIAL_DOMAIN_TRUST_PRIOR_CACHE = out
        return out
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        _OFFICIAL_DOMAIN_TRUST_PRIOR_CACHE = out
        return out
    root_domains = set()
    for d in (payload.get("root_domains") or []):
        ds = str(d or "").strip().lower()
        if ds:
            root_domains.add(ds)
    canonical_subdomains: Dict[str, List[str]] = {}
    raw_subs = payload.get("canonical_subdomains") or {}
    if isinstance(raw_subs, dict):
        for k, v in raw_subs.items():
            kd = str(k or "").strip().lower()
            if not kd:
                continue
            vals = [str(x or "").strip().lower() for x in (v or []) if str(x or "").strip()]
            canonical_subdomains[kd] = vals
    out["root_domains"] = root_domains
    out["canonical_subdomains"] = canonical_subdomains
    _OFFICIAL_DOMAIN_TRUST_PRIOR_CACHE = out
    return out


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
    title_s = str(cap.get("title") or "")
    visible_s = str(cap.get("visible_text_sample") or "")
    security_block_page_detected = bool(cap.get("security_block_page_detected"))
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
        creator_ok, creator_brand, creator_reasons = _creator_platform_host_candidate(
            final_host=final_host,
            final_reg=final_reg,
            title=title_s,
            visible_text=visible_s,
        )
        if creator_ok and str(hp.get("host_identity_class") or "") != "suspicious_host_pattern":
            inactive_markers = ("404", "not found", "page not found", "site unavailable", "unavailable", "not published")
            creator_inactive = any(m in f"{title_s.lower()}\n{visible_s.lower()}" for m in inactive_markers)
            context_type = "creator_platform_404_or_inactive" if creator_inactive else "platform_hosted_legitimate_candidate"
            platform_name = final_reg
            reasons.extend([f"creator_platform:{r}" for r in creator_reasons])
            reasons.append("Creator-platform host/path appears entity-consistent; treat as legitimacy candidate.")
        return {
            "platform_context_type": context_type,
            "platform_name": platform_name,
            "platform_context_reasons": reasons,
            "oauth_providers_detected": sorted(oauth_found),
            "creator_platform_brand": creator_brand if creator_ok else None,
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
        "creator_platform_brand": None,
        "platform_context_blockers": blockers,
    }
