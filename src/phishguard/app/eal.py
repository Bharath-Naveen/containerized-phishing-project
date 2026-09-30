"""Evidence Adjudication Layer: weighs phishing, legitimacy and ambiguity signals and picks the final 3-way verdict.

Split out of the former analyze_dashboard.py in rebuild Phase 2 (code moved unchanged).
"""

from __future__ import annotations
import html as html_lib
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse
from phishguard.features.brand_signals import host_on_official_brand_apex
from .verdict_policy import Verdict3WayConfig, verdict_3way
from .domain_utils import (
    _IMPERSONATION_BRAND_TOKENS,
    _contains_major_brand_token,
    _hostname,
    _is_strong_brand_domain_mismatch,
)
from .registries import (
    _load_official_domain_trust_prior_registry,
)


_EMAIL_IN_URL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def _victim_email_in_url(url: str) -> bool:
    """True when an email address appears in the query string or fragment (not the host or path)."""
    try:
        parts = urlparse(html_lib.unescape(unquote(url)))
    except ValueError:
        return False
    return bool(_EMAIL_IN_URL.search(f"{parts.query}#{parts.fragment}"))


def _apply_evidence_adjudication_layer(
    verdict: Dict[str, Any],
    *,
    ml: Dict[str, Any],
    layer2_capture: Optional[Dict[str, Any]],
    html_structure_summary: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    html_structure_risk: Optional[float],
    html_dom_risk: Optional[float],
    html_dom_enrichment: Optional[Dict[str, Any]] = None,
    behavior_signals: Optional[Dict[str, Any]] = None,
    host_path_reasoning: Optional[Dict[str, Any]],
    platform_context: Optional[Dict[str, Any]],
    trust_blockers: List[str],
    hosting_trust: Optional[Dict[str, Any]],
    legitimacy_bundle: Optional[Dict[str, Any]],
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    out = dict(verdict)
    cap = layer2_capture or {}
    hs = html_structure_summary or {}
    dom = html_dom_summary or {}
    enrich = html_dom_enrichment or {}
    beh = behavior_signals or {}
    hp = host_path_reasoning or {}
    pctx = platform_context or {}
    trust = hosting_trust or {}
    bundle = legitimacy_bundle or {}
    reasons = list(out.get("reasons") or [])
    final_url = str(cap.get("final_url") or "")
    final_host = _hostname(final_url)
    final_reg = str(cap.get("final_registered_domain") or "").lower()
    security_block_page_detected = bool(cap.get("security_block_page_detected"))
    host_identity = str(hp.get("host_identity_class") or "")
    host_legit_conf = str(hp.get("host_legitimacy_confidence") or "")
    no_phishing_guard = bool(out.get("no_phishing_evidence_guard"))
    capture_failed = bool(cap.get("capture_failed"))
    pctx_type = str(pctx.get("platform_context_type") or "")
    platform_hosted_candidate = pctx_type in {"platform_hosted_legitimate_candidate", "creator_platform_404_or_inactive"}
    title_l = str(cap.get("title") or "").lower()
    vis_l = str(cap.get("visible_text_sample") or "").lower()
    platform_404_or_inactive = bool(
        platform_hosted_candidate
        and any(x in f"{title_l}\n{vis_l}" for x in ("404", "not found", "page not found", "site unavailable", "unavailable", "not published"))
    )
    path_l = (urlparse(final_url).path or "").lower() if final_url else ""
    major_brand_impersonation_surface = bool(
        _contains_major_brand_token(final_host)
        or _contains_major_brand_token(path_l)
        or _contains_major_brand_token(title_l)
    )
    no_password_capture = int(hs.get("password_input_count") or 0) == 0
    same_domain_or_platform_forms = int(dom.get("form_action_external_domain_count") or 0) == 0
    no_suspicious_behavior = bool(
        not beh.get("network_exfiltration_suspected")
        and not beh.get("js_dynamic_form_injection_detected")
        and not beh.get("js_anti_debugging_detected")
        and not (beh.get("js_suspicious_fetch_domains") or beh.get("js_suspicious_redirect_domains"))
        and float(beh.get("js_obfuscation_score") or 0.0) < 0.35
    )
    uses_https = bool(cap.get("uses_https"))
    tls_or_cert_error_detected = bool(cap.get("tls_or_cert_error_detected"))
    insecure_scheme_detected = bool(cap.get("insecure_scheme_detected"))
    safe_platform_404_candidate = bool(
        platform_404_or_inactive
        and same_domain_or_platform_forms
        and no_password_capture
        and no_suspicious_behavior
        and not major_brand_impersonation_surface
        and not security_block_page_detected
    )
    creator_platform_uncertainty_cap = bool(
        pctx_type == "platform_hosted_legitimate_candidate"
        and same_domain_or_platform_forms
        and no_password_capture
        and no_suspicious_behavior
        and not major_brand_impersonation_surface
        and not security_block_page_detected
    )
    input_reg = str(cap.get("input_registered_domain") or "")
    same_registrable_domain = bool(input_reg and final_reg and input_reg == final_reg)
    brand_domain_coherence_match = bool(cap.get("brand_domain_coherence_match"))
    weak_or_no_brand_mismatch = str(cap.get("brand_domain_mismatch_strength") or "none").lower() in {"none", "weak"} or not bool(cap.get("brand_domain_mismatch"))
    coherence_clean_context = bool(
        brand_domain_coherence_match
        and same_registrable_domain
        and weak_or_no_brand_mismatch
        and int(hs.get("password_input_count") or 0) == 0
        and int(dom.get("form_action_external_domain_count") or 0) == 0
        and not bool(cap.get("password_input_external_action"))
        and not bool(beh.get("network_exfiltration_suspected"))
        and not security_block_page_detected
        and not bool(cap.get("final_domain_is_free_hosting"))
        and pctx_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
    )
    auth_path_like = any(t in path_l for t in ("login", "signin", "sign-in", "auth", "account", "session", "sessions"))
    first_party_login_like = bool(
        str(dom.get("page_family") or "") == "auth_login_recovery"
        or auth_path_like
        or int(hs.get("password_input_count") or 0) > 0
    )
    html_missing_reason = str(enrich.get("html_capture_missing_reason") or "").strip().lower()
    html_dom_unavailable = bool(
        capture_failed
        or html_missing_reason in {"html_not_available", "html_parse_partial"}
        or (html_structure_risk is None)
        or (html_dom_risk is None)
        or not bool(hs)
        or not bool(dom)
    )
    first_party_auth_flow_cap = bool(
        first_party_login_like
        and same_registrable_domain
        and (not capture_failed)
        and (not html_dom_unavailable)
        and same_domain_or_platform_forms
        and not bool(dom.get("suspicious_credential_collection_pattern") or dom.get("login_harvester_pattern"))
        and not bool(cap.get("brand_domain_mismatch"))
        and not bool(cap.get("final_domain_is_free_hosting"))
        and pctx_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
        and not bool(beh.get("network_exfiltration_suspected"))
        and no_suspicious_behavior
        and not security_block_page_detected
    )
    official_domain_prior_registry = _load_official_domain_trust_prior_registry()
    official_root_domains = set(official_domain_prior_registry.get("root_domains") or set())
    official_domain_prior_candidate = bool(
        final_reg
        and final_reg in official_root_domains
        and same_registrable_domain
        and not bool(cap.get("brand_domain_mismatch"))
        and int(dom.get("form_action_external_domain_count") or 0) == 0
        and not security_block_page_detected
        and not bool(beh.get("network_exfiltration_suspected"))
        and no_suspicious_behavior
        and pctx_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
        and not bool(cap.get("final_domain_is_free_hosting"))
    )

    try:
        p_cal = ml.get("phish_proba_calibrated")
        p_raw = ml.get("phish_proba")
        p = float(p_cal if p_cal is not None else p_raw)
    except Exception:
        p = 0.0
    agr = ml.get("model_agreement") if isinstance(ml.get("model_agreement"), dict) else {}
    cons = str(agr.get("ml_consensus") or "")

    # Coherent first-party page (rule tuning 2026-09, docs/rebuild/TUNING_PLAN.md): the brand the page shows
    # matches the domain it is served from, the visit stayed on that registered domain, transport is valid
    # HTTPS, and nothing posts or leaks off-site. Real sign-in flows (JS-submitted login forms, SSO
    # redirects) look like this; the phishing pages the two blockers below catch did not.
    coherent_first_party = bool(
        same_registrable_domain
        and bool(cap.get("brand_domain_coherence_match"))
        # Other brands mentioned on the page (Apple Pay on a bank page) can raise a "mismatch" even when the
        # title names the site's own domain. That is tolerated only when no brand sits in the host or
        # path and the host does not carry a major brand name unless it is that brand's official domain.
        and (
            str(cap.get("brand_domain_mismatch_strength") or "none").lower() in {"none", "weak"}
            or not bool(cap.get("brand_domain_mismatch"))
            or (
                not bool(cap.get("host_path_brand_context"))
                and not bool(cap.get("brand_in_subdomain_or_path_but_not_registered_domain"))
                and (not _contains_major_brand_token(final_host) or bool(host_on_official_brand_apex(final_host)))
            )
        )
        and uses_https
        and not tls_or_cert_error_detected
        and not insecure_scheme_detected
        and not bool(cap.get("password_input_external_action"))
        and not bool(beh.get("network_exfiltration_suspected"))
        and not security_block_page_detected
        and not bool(cap.get("final_domain_is_free_hosting"))
        and pctx_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
    )

    coherent_js_login = False
    hard_blockers: List[str] = []
    if int(dom.get("form_action_external_domain_count") or 0) > 0 and int(hs.get("password_input_count") or 0) > 0:
        hard_blockers.append("cross_domain_credential_form_action")
    if bool(pctx.get("platform_context_type") == "cloud_hosted_brand_impersonation") or bool(out.get("cloud_hosted_brand_impersonation")):
        hard_blockers.append("cloud_hosted_brand_impersonation")
    if bool(dom.get("login_harvester_pattern")):
        hard_blockers.append("credential_harvesting_pattern")
    elif bool(dom.get("suspicious_credential_collection_pattern")):
        # A password form with an empty action is how most modern sites submit (JavaScript), so on a
        # coherent first-party page it is context, not proof.
        if coherent_first_party:
            coherent_js_login = True
        else:
            hard_blockers.append("credential_harvesting_pattern")
    official_like_host = bool(host_on_official_brand_apex(final_host)) or host_identity == "official_brand_auth"
    same_reg_domain = bool(
        str(cap.get("input_registered_domain") or "")
        and str(cap.get("input_registered_domain") or "") == str(cap.get("final_registered_domain") or "")
    )
    no_brand_mismatch = not bool(cap.get("brand_domain_mismatch"))
    same_domain_forms = int(dom.get("form_action_external_domain_count") or 0) == 0
    ml_not_high_or_strong_legit = cons == "strong_legitimate" or p < 0.70
    weak_or_no_brand_mismatch = str(cap.get("brand_domain_mismatch_strength") or "none").lower() in {"none", "weak"} or no_brand_mismatch
    no_credential_capture = bool(int(hs.get("password_input_count") or 0) == 0 and int(hs.get("email_input_count") or 0) == 0)
    no_password_external_action = not bool(cap.get("password_input_external_action"))
    brand_domain_coherence_match = bool(cap.get("brand_domain_coherence_match"))
    coherent_brand_host_identity_candidate = bool(
        brand_domain_coherence_match
        and same_reg_domain
        and weak_or_no_brand_mismatch
        and not bool(cap.get("final_domain_is_free_hosting"))
        and pctx_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
    )
    rich_nav_footer_support = bool(
        int(hs.get("nav_link_count") or 0) >= 4
        and int(hs.get("footer_link_count") or 0) >= 2
        and bool(hs.get("has_support_help_links"))
    )
    wrapper_clean_official_or_same_domain_safe = bool(
        same_reg_domain
        and no_credential_capture
        and same_domain_forms
        and no_password_external_action
        and not bool(beh.get("network_exfiltration_suspected"))
        and not security_block_page_detected
        and weak_or_no_brand_mismatch
        and not bool(cap.get("final_domain_is_free_hosting"))
        and pctx_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
        and (rich_nav_footer_support or official_domain_prior_candidate or coherent_brand_host_identity_candidate)
    )
    wrapper_official_authwall_safe = bool(
        official_like_host
        and same_reg_domain
        and no_brand_mismatch
        and same_domain_forms
        and ml_not_high_or_strong_legit
        and host_legit_conf == "high"
    )
    official_domain_family = bool(bundle.get("official_domain_family"))
    official_auth_same_domain_safe = bool(
        same_reg_domain
        and (
            host_identity in {"official_brand_auth", "official_brand_apex", "official_brand_host"}
            or official_like_host
            or official_domain_family
        )
        and same_domain_forms
        and no_password_external_action
        and not bool(beh.get("network_exfiltration_suspected"))
        and not security_block_page_detected
        and not bool(cap.get("final_domain_is_free_hosting"))
        and pctx_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
        and (
            bool(cap.get("brand_domain_coherence_match"))
            or official_domain_family
        )
    )
    if bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern")):
        if (wrapper_official_authwall_safe or wrapper_clean_official_or_same_domain_safe
                or official_auth_same_domain_safe or coherent_first_party):
            # Official first-party authwall/login wrappers are ambiguity context, not hard blockers.
            pass
        else:
            hard_blockers.append("wrapper_or_interstitial_redirect_pattern")
    if bool(cap.get("contains_punycode") or cap.get("contains_non_ascii")) and int(hs.get("password_input_count") or 0) > 0:
        hard_blockers.append("punycode_or_non_ascii_with_credential_context")
    if bool(beh.get("network_exfiltration_suspected")) and (
        int(hs.get("password_input_count") or 0) > 0 or int(hs.get("form_count") or 0) > 0
    ):
        hard_blockers.append("network_exfiltration_with_credential_context")
    # Victim email pre-filled in the submitted URL (query or fragment), a common phishing-link pattern,
    # counted only when the URL model also leans phishing and the page is not a coherent first-party page.
    if _victim_email_in_url(str(ml.get("canonical_url") or "")) and p >= 0.50 and not coherent_first_party:
        hard_blockers.append("victim_email_prefilled_in_url")

    phish_signals: List[str] = []
    legit_signals: List[str] = []
    amb_signals: List[str] = []
    phish_score = 0.0
    legit_score = 0.0

    if p >= 0.85:
        phish_score += 0.35
        phish_signals.append("high_ml_score")
    elif p >= 0.70:
        phish_score += 0.20
        phish_signals.append("elevated_ml_score")
    elif p >= 0.50:
        amb_signals.append("moderate_ml_score")

    if cons == "strong_phishing":
        phish_score += 0.12
        phish_signals.append("ml_consensus_strong_phishing")
    elif cons == "strong_legitimate":
        legit_score += 0.12
        legit_signals.append("ml_consensus_strong_legitimate")
    if cons == "split":
        amb_signals.append("ml_consensus_split")
        if not hard_blockers and phish_score < 0.40:
            legit_score += 0.06
    elif cons == "boosted_only":
        amb_signals.append("boosted_models_only_flag_phishing")
    spr = agr.get("ml_prob_spread")
    if isinstance(spr, (int, float)) and float(spr) > 0.4:
        amb_signals.append("high_model_probability_spread")
    if wrapper_official_authwall_safe and bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern")):
        amb_signals.append("official_authwall_wrapper_pattern")
    if official_auth_same_domain_safe and bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern")):
        amb_signals.append("official_authwall_wrapper_pattern")
    if wrapper_clean_official_or_same_domain_safe and bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern")):
        amb_signals.append("official_content_wrapper_pattern")
        if coherent_brand_host_identity_candidate:
            amb_signals.append("coherent_brand_wrapper_pattern")
    if coherent_js_login:
        amb_signals.append("first_party_script_submitted_login")
    if bool(beh.get("js_dynamic_form_injection_detected")):
        phish_score += 0.20
        phish_signals.append("js_dynamic_form_injection_detected")
    if bool(beh.get("js_anti_debugging_detected")):
        phish_score += 0.15
        phish_signals.append("js_anti_debugging_detected")
    js_obf = float(beh.get("js_obfuscation_score") or 0.0)
    js_obf_signals = set(beh.get("js_obfuscation_signals") or [])
    has_auth_context = bool(
        int(hs.get("password_input_count") or 0) > 0
        or int(hs.get("form_count") or 0) > 0
        or str(dom.get("page_family") or "") == "auth_login_recovery"
    )
    if has_auth_context and (js_obf >= 0.5 or {"eval_atob_pattern", "eval_unescape_pattern"} & js_obf_signals):
        phish_score += 0.15
        phish_signals.append("high_js_obfuscation_auth_context")
    elif js_obf >= 0.35:
        phish_score += 0.08
        phish_signals.append("elevated_js_obfuscation")
    if (beh.get("js_suspicious_fetch_domains") or beh.get("js_suspicious_redirect_domains")) and has_auth_context:
        phish_score += 0.18
        phish_signals.append("js_unrelated_network_calls_with_auth_context")
    elif bool(beh.get("network_unrelated_domains")):
        amb_signals.append("unrelated_third_party_network_domains")
        if len(beh.get("network_unrelated_domains") or []) >= 5 and pctx_type not in {
            "official_platform_domain",
            "official_platform_login",
            "platform_hosted_legitimate_candidate",
            "creator_platform_404_or_inactive",
        }:
            phish_score += 0.08
            phish_signals.append("many_unrelated_third_party_domains")
    if bool(beh.get("behavior_analysis_unavailable")):
        amb_signals.append("behavior_analysis_unavailable")
    if (
        (tls_or_cert_error_detected or insecure_scheme_detected)
        and (
            int(hs.get("password_input_count") or 0) > 0
            or str(dom.get("page_family") or "") in {"auth_login_recovery", "checkout_payment"}
            or any(t in (urlparse(final_url).path or "").lower() for t in ("login", "signin", "auth", "verify", "checkout", "payment"))
        )
    ):
        phish_score += 0.18
        phish_signals.append("insecure_auth_surface")
    if first_party_auth_flow_cap and (p >= 0.70 or cons in {"strong_phishing", "boosted_only"}):
        amb_signals.append("first_party_login_ml_disagreement")
    if platform_404_or_inactive:
        amb_signals.append("platform_404_or_inactive")
    if security_block_page_detected:
        amb_signals.append("security_block_page_observed")
        phish_score += 0.18
        phish_signals.append("security_vendor_blocked_as_malicious")

    if host_identity == "suspicious_host_pattern":
        phish_score += 0.30
        phish_signals.append("suspicious_host_pattern")
        if str(pctx.get("platform_context_type") or "") == "user_hosted_subdomain":
            phish_score += 0.30
            phish_signals.append("suspicious_user_hosted_subdomain")
    strong_brand_mismatch = _is_strong_brand_domain_mismatch(cap, dom)
    if official_auth_same_domain_safe:
        strong_brand_mismatch = False
    if strong_brand_mismatch and pctx_type in {"official_platform_domain", "official_platform_login"}:
        # On official platform domains, only trust/auth/host-path impersonation should remain strong.
        trust_or_host_impersonation = bool(
            cap.get("trust_surface_brand_context")
            or cap.get("host_path_brand_context")
            or cap.get("auth_payment_brand_context")
            or _contains_major_brand_token(final_host)
            or _contains_major_brand_token((urlparse(final_url).path or "").lower() if final_url else "")
            or _contains_major_brand_token(str(cap.get("title") or ""))
            or _contains_major_brand_token(str(hs.get("h1_text") or ""))
        )
        if not trust_or_host_impersonation:
            strong_brand_mismatch = False
    deceptive_url_structure = bool(
        cap.get("host_path_brand_context")
        or cap.get("brand_in_subdomain_or_path_but_not_registered_domain")
        or (host_identity == "suspicious_host_pattern")
        or bool(cap.get("final_domain_is_free_hosting"))
    )
    credential_harvest_context = bool(
        int(hs.get("password_input_count") or 0) > 0
        or bool(dom.get("suspicious_credential_collection_pattern"))
        or bool(dom.get("login_harvester_pattern"))
        or int(dom.get("form_action_external_domain_count") or 0) > 0
    )
    if strong_brand_mismatch and (credential_harvest_context or deceptive_url_structure):
        phish_score += 0.30
        phish_signals.append("strong_brand_domain_mismatch")
    if strong_brand_mismatch and pctx_type in {"platform_hosted_legitimate_candidate", "creator_platform_404_or_inactive"}:
        creator_safe_context = bool(
            int(hs.get("password_input_count") or 0) == 0
            and int(dom.get("form_action_external_domain_count") or 0) == 0
            and not bool(beh.get("network_exfiltration_suspected"))
            and not _contains_major_brand_token(final_host)
            and not _contains_major_brand_token((urlparse(final_url).path or "").lower() if final_url else "")
            and not _contains_major_brand_token(str(cap.get("title") or ""))
            and not _contains_major_brand_token(str(hs.get("h1_text") or ""))
        )
        creator_404_or_newsletter = bool(
            any(x in f"{str(cap.get('title') or '').lower()}\n{str(cap.get('visible_text_sample') or '').lower()}\n{(urlparse(final_url).path or '').lower() if final_url else ''}" for x in ("404", "not found", "page not found", "newsletter", "/404"))
        )
        if creator_safe_context and creator_404_or_newsletter:
            # Demote over-aggressive mismatch on benign creator platform 404/newsletter context.
            phish_score = max(0.0, phish_score - 0.30)
            try:
                phish_signals.remove("strong_brand_domain_mismatch")
            except ValueError:
                pass
    if bool(out.get("dormant_phishing_infra_detected")):
        phish_score += 0.40
        phish_signals.append("dormant_phishing_infrastructure")
    if bool(cap.get("capture_failure_suspicious")):
        phish_score += 0.20
        phish_signals.append("capture_failure_suspicious")
    if bool(cap.get("final_domain_is_free_hosting")) and bool(_is_strong_brand_domain_mismatch(cap, dom)):
        phish_score += 0.25
        phish_signals.append("free_hosted_brand_impersonation")
    if (
        str(dom.get("page_family") or "") == "auth_login_recovery"
        and (not official_auth_same_domain_safe)
        and str(pctx.get("platform_context_type") or "") not in {
        "official_platform_domain",
        "official_platform_login",
        }
    ):
        phish_score += 0.20
        phish_signals.append("auth_context_on_non_official_domain")

    # Strong phishing evidence: brand token in subdomain not matching registrable family + suspicious/low host legitimacy.
    brand_subdomain_impersonation = False
    if final_host and final_reg and final_host.endswith("." + final_reg):
        sub_label = final_host[: -(len(final_reg) + 1)].lower()
        for tok in _IMPERSONATION_BRAND_TOKENS:
            t = str(tok or "").strip().lower()
            if len(t) < 3:
                continue
            if t in sub_label and t not in final_reg:
                brand_subdomain_impersonation = True
                break
    if brand_subdomain_impersonation and (host_identity == "suspicious_host_pattern" or host_legit_conf == "low"):
        phish_score += 0.35
        phish_signals.append("brand_subdomain_impersonation")

    if (
        str(cap.get("input_registered_domain") or "")
        and str(cap.get("input_registered_domain") or "") == str(cap.get("final_registered_domain") or "")
        and host_identity != "suspicious_host_pattern"
    ):
        legit_score += 0.20
        legit_signals.append("same_domain_consistency")
    elif str(cap.get("input_registered_domain") or "") and str(cap.get("input_registered_domain") or "") == str(
        cap.get("final_registered_domain") or ""
    ):
        amb_signals.append("same_domain_on_suspicious_registrable")
    if pctx_type in {"official_platform_domain", "official_platform_login"}:
        legit_score += 0.30
        legit_signals.append("official_platform_domain")
    if platform_hosted_candidate and str(cap.get("brand_domain_mismatch_strength") or "none") in {"none", "weak"}:
        legit_score += 0.20
        legit_signals.append("platform_hosted_brand_consistency")
    if first_party_auth_flow_cap:
        legit_score += 0.18
        legit_signals.append("first_party_auth_flow_consistency")
    if official_auth_same_domain_safe and "first_party_auth_flow_consistency" not in legit_signals:
        legit_score += 0.18
        legit_signals.append("first_party_auth_flow_consistency")
    if coherence_clean_context:
        legit_score += 0.18
        legit_signals.append("brand_domain_coherence")
        if not official_domain_prior_candidate:
            legit_score += 0.10
            legit_signals.append("coherent_brand_host_identity")
    if uses_https and not tls_or_cert_error_detected and not insecure_scheme_detected:
        legit_score += 0.05
        legit_signals.append("valid_https_transport")
    if official_domain_prior_candidate:
        legit_score += 0.14
        legit_signals.append("official_domain_trust_prior")
    hts = str(trust.get("hosting_trust_status") or "")
    if hts == "hosting_trust_verified":
        legit_score += 0.35
        legit_signals.append("hosting_trust_verified")
    elif hts == "hosting_trust_partial":
        legit_score += 0.25
        legit_signals.append("hosting_trust_partial")
    if (not security_block_page_detected) and (not html_dom_unavailable) and int(hs.get("password_input_count") or 0) == 0 and not bool(dom.get("suspicious_credential_collection_pattern")):
        legit_score += 0.25
        legit_signals.append("no_credential_capture")
    if (not security_block_page_detected) and (not html_dom_unavailable) and int(dom.get("form_action_external_domain_count") or 0) == 0:
        legit_score += 0.20
        legit_signals.append("no_cross_domain_forms")
    free_hosted_brand_imp = bool(cap.get("final_domain_is_free_hosting")) and bool(_is_strong_brand_domain_mismatch(cap, dom))
    cloud_hosted_brand_imp = bool(
        pctx.get("platform_context_type") == "cloud_hosted_brand_impersonation"
    ) or bool(out.get("cloud_hosted_brand_impersonation"))
    impersonation_hosting_context = bool(free_hosted_brand_imp or cloud_hosted_brand_imp)

    if (not security_block_page_detected) and (not impersonation_hosting_context) and (
        bool(dom.get("content_rich_profile"))
        or str(dom.get("page_family") or "") in {
        "article_news",
        "content_feed_forum_aggregator",
        "public_docs_or_reference",
        "generic_landing",
        }
    ):
        legit_score += 0.20
        legit_signals.append("content_rich_page")
    if (not security_block_page_detected) and (not impersonation_hosting_context) and (float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else 1.0) <= 0.20 and (
        float(html_structure_risk) if isinstance(html_structure_risk, (int, float)) else 1.0
    ) <= 0.30:
        legit_score += 0.20
        legit_signals.append("low_html_dom_risk")
    if bool(hs.get("has_support_help_links")) and int(hs.get("nav_link_count") or 0) >= 4 and int(hs.get("footer_link_count") or 0) >= 2:
        legit_score += 0.15
        legit_signals.append("rich_nav_footer_support")

    if bool(cap.get("brand_domain_mismatch")) and not _is_strong_brand_domain_mismatch(cap, dom):
        amb_signals.append("weak_brand_mismatch")
    if int(cap.get("encoded_char_count") or 0) > 0 or int(cap.get("suspicious_keyword_count") or 0) > 0:
        amb_signals.append("url_tracking_or_length_noise")
    if bool(out.get("inactive_site_detected")):
        amb_signals.append("inactive_site_context")
    if html_dom_unavailable:
        amb_signals.append("html_dom_unavailable")
    if phish_score >= 0.45 and legit_score >= 0.45:
        amb_signals.append("ml_structural_disagreement")

    evidence_conf = int(bool(phish_signals)) + int(bool(legit_signals)) + int(bool(amb_signals))
    freehost_mismatch_consensus_escalation = bool(
        "free_hosted_brand_impersonation" in phish_signals
        and "strong_brand_domain_mismatch" in phish_signals
        and "ml_consensus_strong_phishing" in phish_signals
    )
    trusted_official_context = bool(
        pctx_type in {"official_platform_domain", "official_platform_login"}
        or host_on_official_brand_apex(final_host)
        or (
            str(trust.get("hosting_trust_status") or "") in {"hosting_trust_verified", "hosting_trust_partial"}
            and host_legit_conf == "high"
            and host_identity != "suspicious_host_pattern"
        )
    )
    creator_legitimate_context = bool(
        pctx_type in {"platform_hosted_legitimate_candidate", "creator_platform_404_or_inactive"}
    )
    missing_evidence_high_risk = bool(
        (capture_failed or html_dom_unavailable)
        and (
            bool(beh.get("behavior_analysis_unavailable"))
            or html_missing_reason in {"html_not_available", "html_parse_partial"}
        )
        and (p >= 0.90 or cons == "strong_phishing")
        and (
            pctx_type == "user_hosted_subdomain"
            or host_legit_conf == "low"
            or host_identity == "suspicious_host_pattern"
            or bool(cap.get("capture_failure_suspicious"))
        )
        and (not trusted_official_context)
        and (not creator_legitimate_context)
        and (not security_block_page_detected)
    )
    if missing_evidence_high_risk:
        phish_score += 0.20
        phish_signals.append("high_risk_missing_evidence")
    security_block_escalation = bool(
        security_block_page_detected
        and (
            "ml_consensus_strong_phishing" in phish_signals
            or "high_ml_score" in phish_signals
            or "strong_brand_domain_mismatch" in phish_signals
        )
    )
    official_domain_conflict_relief = bool(
        official_domain_prior_candidate
        and (not hard_blockers)
        and (not security_block_page_detected)
        and phish_score >= 0.70
        and legit_score >= 0.35
    )
    official_domain_ml_overconfidence = bool(
        official_domain_prior_candidate
        and (not capture_failed)
        and (not html_dom_unavailable)
        and same_domain_or_platform_forms
        and not bool(cap.get("brand_domain_mismatch"))
        and not bool(beh.get("network_exfiltration_suspected"))
        and not security_block_page_detected
        and (not hard_blockers)
        and (
            bool(dom.get("content_rich_profile"))
            or str(dom.get("page_family") or "") in {"article_news", "content_feed_forum_aggregator", "public_docs_or_reference", "generic_landing"}
        )
        and (float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else 1.0) <= 0.20
        and (float(html_structure_risk) if isinstance(html_structure_risk, (int, float)) else 1.0) <= 0.30
        and (p >= 0.90 or cons == "strong_phishing")
    )
    ml_brand_coherence_disagreement = bool(
        coherence_clean_context
        and (not hard_blockers)
        and (
            bool(dom.get("content_rich_profile"))
            or str(dom.get("page_family") or "") in {"article_news", "content_feed_forum_aggregator", "public_docs_or_reference", "generic_landing"}
        )
        and (float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else 1.0) <= 0.20
        and (float(html_structure_risk) if isinstance(html_structure_risk, (int, float)) else 1.0) <= 0.30
        and (p >= 0.90 or cons == "strong_phishing")
    )
    if official_domain_ml_overconfidence:
        amb_signals.append("official_domain_ml_overconfidence_suspected")
    if ml_brand_coherence_disagreement:
        amb_signals.append("ml_brand_coherence_disagreement")
    auth_page_like = str(dom.get("page_family") or "") == "auth_login_recovery" or first_party_login_like
    ml_overconfidence_relaxed_due_to_strong_legitimacy = bool(
        official_domain_prior_candidate
        and coherence_clean_context
        and same_registrable_domain
        and no_credential_capture
        and same_domain_or_platform_forms
        and not bool(cap.get("password_input_external_action"))
        and not bool(beh.get("network_exfiltration_suspected"))
        and not security_block_page_detected
        and not hard_blockers
        and not auth_page_like
        and cons == "strong_phishing"
        and legit_score >= 0.90
    )
    if ml_overconfidence_relaxed_due_to_strong_legitimacy:
        amb_signals.append("ml_overconfidence_relaxed_due_to_strong_legitimacy")
    official_domain_clean_promotion = bool(
        official_domain_prior_candidate
        and (cons == "strong_legitimate" or p < 0.50 or no_phishing_guard)
        and legit_score >= 0.60
        and phish_score < 0.70
        and not auth_page_like
    )

    if no_phishing_guard and not hard_blockers:
        # Deterministic precedence: no-phishing-evidence guard cannot end as phishing without hard blockers.
        final_label = "likely_legitimate" if legit_score >= 0.45 and phish_score < 0.70 else "uncertain"
    elif hard_blockers:
        final_label = "likely_phishing"
    elif missing_evidence_high_risk:
        final_label = "likely_phishing"
        reasons.append("Missing live/HTML/DOM evidence with high ML and suspicious host context elevates risk.")
    elif first_party_auth_flow_cap:
        # First-party auth flow on same registrable domain with no exfiltration/impersonation: keep high-ML disagreement conservative.
        final_label = "uncertain"
    elif ml_overconfidence_relaxed_due_to_strong_legitimacy:
        final_label = "likely_legitimate"
    elif ml_brand_coherence_disagreement:
        # Strong title/header-brand to host coherence with clean live evidence should cap high-ML disagreement.
        final_label = "uncertain"
    elif official_domain_ml_overconfidence:
        final_label = "uncertain"
    elif official_domain_conflict_relief:
        final_label = "uncertain"
    elif creator_platform_uncertainty_cap:
        # Cap creator-platform candidate pages at uncertain when no credential/exfiltration/impersonation hard evidence exists.
        final_label = "uncertain"
    elif security_block_escalation:
        final_label = "likely_phishing"
    elif freehost_mismatch_consensus_escalation:
        final_label = "likely_phishing"
    elif capture_failed and host_identity == "suspicious_host_pattern":
        if p >= 0.95:
            final_label = "likely_phishing"
        elif p >= 0.80 and brand_subdomain_impersonation:
            final_label = "likely_phishing"
        elif brand_subdomain_impersonation and host_legit_conf == "low":
            final_label = "likely_phishing"
        else:
            final_label = "uncertain"
    elif str(pctx.get("platform_context_type") or "") == "official_platform_login" and p >= 0.85:
        # Keep high-ML official login disagreement conservative.
        final_label = "uncertain"
    elif capture_failed and html_dom_unavailable:
        # Evidence-gap condition: absence-based structural legitimacy cannot establish safety.
        final_label = "uncertain"
    elif (
        str(pctx.get("platform_context_type") or "") == "user_hosted_subdomain"
        and str(hp.get("host_identity_class") or "") == "suspicious_host_pattern"
        and phish_score >= 0.60
    ):
        final_label = "likely_phishing"
    elif safe_platform_404_candidate and not hard_blockers and "strong_brand_domain_mismatch" not in phish_signals:
        # Known creator-platform inactive pages should not be convicted by ML/model-agreement alone.
        final_label = "uncertain"
    elif phish_score >= 0.70 and evidence_conf >= 2:
        final_label = "likely_phishing"
    elif bool(out.get("inactive_site_detected")):
        final_label = "uncertain"
    elif official_domain_clean_promotion:
        final_label = "likely_legitimate"
    elif legit_score >= 0.70 and phish_score < 0.45:
        final_label = "likely_legitimate"
    else:
        final_label = "uncertain"
    if security_block_page_detected and final_label == "likely_legitimate":
        final_label = "uncertain"

    out["evidence_adjudication_applied"] = True
    out["evidence_adjudication_verdict"] = final_label
    out["evidence_phishing_score"] = round(float(phish_score), 4)
    out["evidence_legitimacy_score"] = round(float(legit_score), 4)
    out["evidence_confidence"] = int(evidence_conf)
    out["evidence_hard_blockers"] = hard_blockers
    out["evidence_phishing_signals"] = phish_signals
    out["evidence_legitimacy_signals"] = legit_signals
    out["evidence_ambiguity_signals"] = amb_signals
    out["evidence_adjudication_reasons"] = (
        [f"hard_blocker:{b}" for b in hard_blockers[:4]]
        + [f"phishing_signal:{s}" for s in phish_signals[:4]]
        + [f"legitimacy_signal:{s}" for s in legit_signals[:4]]
        + [f"ambiguity_signal:{s}" for s in amb_signals[:3]]
    )
    out["verdict_3way"] = final_label
    out["label"] = final_label
    out["confidence"] = "medium" if final_label != "uncertain" else "low"
    if final_label == "likely_phishing":
        out["combined_score"] = max(float(out.get("combined_score") or 0.0), float((verdict_cfg or Verdict3WayConfig()).combined_high))
    elif final_label == "likely_legitimate":
        out["combined_score"] = min(float(out.get("combined_score") or 0.0), float((verdict_cfg or Verdict3WayConfig()).combined_low) - 1e-3)
    reasons.extend(["Final Evidence Review applied deterministic evidence adjudication."])
    if html_dom_unavailable:
        reasons.extend(
            [
                "Capture/HTML/DOM evidence was unavailable; absence-based safety signals were not credited.",
                "Page could not be validated fully due to missing live/HTML/DOM evidence.",
            ]
        )
    out["reasons"] = reasons
    return out


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
