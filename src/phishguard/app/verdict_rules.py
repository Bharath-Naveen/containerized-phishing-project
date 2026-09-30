"""Score adjustments applied before the EAL: capture-miss safety, capture-failure hardening, official-apex cap, legitimacy rescue, hosting trust, platform context, DNS dampening, ML overconfidence cap.

Split out of the former analyze_dashboard.py in rebuild Phase 2 (code moved unchanged).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from phishguard.features.brand_signals import host_on_official_brand_apex
from phishguard.urls.safe import safe_hostname
from .runtime_config import PipelineConfig
from .verdict_policy import Verdict3WayConfig, verdict_3way
from .domain_utils import (
    _BUILDER_PLATFORM_SUFFIXES,
    _OFFICIAL_BRAND_APEX_PHISH_CAP,
    _detect_cloud_hosted_brand_impersonation,
    _hostname,
    _is_strong_brand_domain_mismatch,
    _safe_subdomain_allowed,
)
from .registries import (
    _load_trusted_domain_registry,
)


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
    same_reg_domain = bool(
        str(cap.get("input_registered_domain") or "")
        and str(cap.get("input_registered_domain") or "") == str(cap.get("final_registered_domain") or "")
    )
    weak_or_no_brand_mismatch = mismatch_strength in {"none", "weak"} or not bool(cap.get("brand_domain_mismatch"))
    brand_domain_coherence_match = bool(cap.get("brand_domain_coherence_match"))
    official_domain_family = bool(legitimacy_bundle.get("official_domain_family"))
    official_like_host = bool(host_on_official_brand_apex(_hostname(str(cap.get("final_url") or "")))) or str(hp.get("host_identity_class") or "") in {
        "official_brand_auth",
        "official_brand_apex",
        "official_brand_host",
    }
    coherent_brand_host_identity_candidate = bool(
        brand_domain_coherence_match
        and same_reg_domain
        and weak_or_no_brand_mismatch
        and not bool(cap.get("final_domain_is_free_hosting"))
        and platform_context_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
    )
    official_auth_same_domain_safe = bool(
        same_reg_domain
        and (official_like_host or official_domain_family)
        and int(dom.get("form_action_external_domain_count") or 0) == 0
        and not bool(cap.get("password_input_external_action"))
        and not bool(cap.get("network_exfiltration_suspected"))
        and not bool(cap.get("security_block_page_detected"))
        and not bool(cap.get("final_domain_is_free_hosting"))
        and platform_context_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
        and (brand_domain_coherence_match or official_domain_family)
    )
    rich_nav_footer_support = bool(
        int(cap.get("nav_link_count") or 0) >= 4
        and int(cap.get("footer_link_count") or 0) >= 2
        and bool(cap.get("has_support_help_links"))
    )
    clean_official_content_wrapper_context = bool(
        same_reg_domain
        and weak_or_no_brand_mismatch
        and int(dom.get("form_action_external_domain_count") or 0) == 0
        and int(cap.get("password_input_count") or 0) == 0
        and not bool(cap.get("password_input_external_action"))
        and not bool(cap.get("network_exfiltration_suspected"))
        and not bool(cap.get("security_block_page_detected"))
        and not bool(cap.get("final_domain_is_free_hosting"))
        and platform_context_type not in {"user_hosted_subdomain", "cloud_hosted_brand_impersonation"}
        and not bool(dom.get("suspicious_credential_collection_pattern") or dom.get("login_harvester_pattern"))
        and (rich_nav_footer_support or bool(dom.get("content_rich_profile")) or coherent_brand_host_identity_candidate)
    )
    blockers: List[str] = []
    if int(dom.get("form_action_external_domain_count") or 0) > 0 or bool(legitimacy_bundle.get("suspicious_form_action_cross_origin")):
        blockers.append("cross_domain_form_action")
    if strong_brand_mismatch and (not suppress_oauth_brand_mismatch) and (not official_auth_same_domain_safe):
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
    if bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern")) and not clean_official_content_wrapper_context and not official_auth_same_domain_safe:
        blockers.append("wrapper_or_interstitial_redirect_pattern")
    if int(dom.get("anchor_strong_mismatch_count") or 0) > 0:
        blockers.append("strong_anchor_domain_mismatch")
    if str(hp.get("host_identity_class") or "") == "suspicious_host_pattern":
        blockers.append("suspicious_host_pattern")
    return blockers


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
