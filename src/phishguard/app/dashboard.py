"""Layer-1 ML + optional reinforcement capture → dashboard JSON (no screenshot-vs-reference flow).

The optional **official-brand apex cap** below is a *temporary UX safety net* only. Primary trust
should be the retrained Layer-1 model and structural brand features in
:mod:`phishguard.features.brand_signals`.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
from phishguard.paths import analysis_dir, ensure_layout
from .capture import capture_url
from .behavior_signals import extract_behavior_signals
from .runtime_config import PipelineConfig
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
from .ml_layer1 import compute_layer1_model_agreement, predict_layer1
from .org_style_signals import dampen_org_style_for_page_family, org_style_from_capture_blob
from .schemas import utc_now_iso
from .verdict_policy import Verdict3WayConfig, verdict_3way
from .domain_utils import (  # noqa: F401  (re-exported: tests and callers import these from dashboard)
    _OFFICIAL_BRAND_APEX_PHISH_CAP,
    _FREE_HOSTING_SUFFIXES,
    _BUILDER_PLATFORM_SUFFIXES,
    _CREATOR_PLATFORM_SUFFIXES,
    _URL_WEAK_KEYWORDS,
    _SUSPICIOUS_HTML_KEYWORDS,
    _IMPERSONATION_BRAND_TOKENS,
    _reg_domain,
    _hostname,
    _is_major_brand_token,
    _contains_major_brand_token,
    _normalize_entity_token,
    _safe_subdomain_allowed,
    _is_strong_brand_domain_mismatch,
    _detect_cloud_hosted_brand_impersonation,
    _creator_platform_host_candidate,
)
from .registries import (  # noqa: F401  (re-exported: tests and callers import these from dashboard)
    _TRUSTED_DOMAIN_REGISTRY_CACHE,
    _TRUSTED_DOMAIN_REGISTRY_CACHE_PATH,
    _PLATFORM_DOMAIN_REGISTRY_CACHE,
    _PLATFORM_DOMAIN_REGISTRY_CACHE_PATH,
    _OFFICIAL_DOMAIN_TRUST_PRIOR_CACHE,
    _REPO_ROOT,
    _OFFICIAL_DOMAINS_JSON_PATH,
    _load_trusted_domain_registry,
    _load_platform_domain_registry,
    _load_official_domain_trust_prior_registry,
    _classify_platform_host_context,
)
from .brand_coherence import (  # noqa: F401  (re-exported: tests and callers import these from dashboard)
    normalize_brand_text,
    tokenize_domain_brand,
    extract_primary_brand_candidates,
    compute_brand_domain_coherence,
)
from .capture_signals import (  # noqa: F401  (re-exported: tests and callers import these from dashboard)
    _FASTTEXT_MIN_TEXT_CHARS,
    _FASTTEXT_MODEL_CACHE,
    _FASTTEXT_MODEL_ERROR,
    _SECURITY_BLOCK_VENDOR_PATTERNS,
    _load_fasttext_language_model,
    _detect_security_block_page,
    _fasttext_language_enrichment,
    _detect_oauth_providers,
    _enrich_capture_and_html_signals,
    _classify_capture_failure_type,
    _capture_failure_plain_reason,
    _merge_capture_failure_fields,
    _finalize_click_probe_diagnostics_on_capture,
)
from .verdict_rules import (  # noqa: F401  (re-exported: tests and callers import these from dashboard)
    _ML_CAPTURE_MISS_SAFETY_REASON_UNCERTAIN,
    _ML_CAPTURE_MISS_SAFETY_REASON_HIGH,
    _trust_anchor_present,
    _html_missing_after_capture_failure,
    _apply_ml_phishing_capture_miss_legitimacy_safety,
    _apply_capture_failure_verdict_hardening,
    _apply_official_brand_apex_cap,
    _verdict_from_scores,
    _apply_legitimacy_rescue_on_verdict,
    _dns_contribution_profile,
    _compute_phishing_blockers,
    _evaluate_hosting_domain_trust,
    _apply_hosting_trust_promotion,
    _apply_untrusted_builder_hosting_downgrade,
    _apply_inactive_site_overlay,
    _apply_platform_context_policy,
    _apply_dns_feature_dominance_dampening,
    _apply_ml_overconfidence_cap,
)
from .eal import (  # noqa: F401  (re-exported: tests and callers import these from dashboard)
    _apply_evidence_adjudication_layer,
    no_phishing_evidence_guard,
    _apply_no_phishing_evidence_override,
)

logger = logging.getLogger(__name__)


def build_dashboard_analysis(
    url: str,
    *,
    reinforcement: bool = True,
    layer1_use_dns: bool = False,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Core analysis dict + evidence gap strings (no file write)."""
    ensure_layout()
    cfg = PipelineConfig.from_env()
    url = (url or "").strip()
    evidence_gaps: List[str] = []
    ml = predict_layer1(url, use_dns=layer1_use_dns)
    ml = _apply_official_brand_apex_cap(ml, url)
    ml["model_agreement"] = compute_layer1_model_agreement(url, ml, use_dns=layer1_use_dns)
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
                    "network_request_urls": list(cap.network_request_urls or []),
                    "uses_https": bool(cap.uses_https),
                    "browser_security_state": cap.browser_security_state,
                    "tls_or_cert_error_detected": bool(cap.tls_or_cert_error_detected),
                    "insecure_scheme_detected": bool(cap.insecure_scheme_detected),
                    "mixed_content_detected": bool(cap.mixed_content_detected),
                    "security_state_reasons": list(cap.security_state_reasons or []),
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

    _ = pre_ml_capture_miss_safety_verdict
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
    behavior_signals = extract_behavior_signals(
        html_path=str(html_path) if html_path else None,
        layer2_capture=capture_for_policy,
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        platform_context_type=str(platform_context.get("platform_context_type") or "unknown"),
    )
    verdict = _apply_evidence_adjudication_layer(
        verdict,
        ml=ml,
        layer2_capture=capture_for_policy,
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        html_structure_risk=html_structure.get("html_structure_risk_score"),
        html_dom_risk=html_dom.get("html_dom_anomaly_risk_score"),
        html_dom_enrichment=layer3_enrichment,
        behavior_signals=behavior_signals,
        host_path_reasoning=host_path_reasoning,
        platform_context=platform_context,
        trust_blockers=trust_blockers,
        hosting_trust=hosting_trust,
        legitimacy_bundle=bundle,
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
        "behavior_signals": behavior_signals,
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
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    out, _ = build_dashboard_analysis(
        url,
        reinforcement=reinforcement,
        layer1_use_dns=layer1_use_dns,
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
    args = ap.parse_args()
    row = analyze_url_dashboard(
        args.url,
        reinforcement=not args.no_reinforcement,
        layer1_use_dns=args.layer1_use_dns,
    )
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
