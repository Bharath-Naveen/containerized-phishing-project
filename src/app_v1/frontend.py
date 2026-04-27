"""Streamlit phishing analysis dashboard (Layer-1 ML + optional reinforcement).

Run from repo root::

    streamlit run src/app_v1/frontend.py

Legacy screenshot UI: ``archive/legacy/frontend_screenshot_legacy.py``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import streamlit as st

logger = logging.getLogger(__name__)


def _run_dashboard_subprocess(
    url: str,
    reinforcement: bool,
    layer1_dns: bool,
    ai_adjudication: bool,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "src.app_v1.analyze_dashboard",
        "--url",
        url,
    ]
    if not reinforcement:
        cmd.append("--no-reinforcement")
    if layer1_dns:
        cmd.append("--layer1-use-dns")
    if not ai_adjudication:
        cmd.append("--no-ai-adjudication")
    env = os.environ.copy()
    sep = os.pathsep
    root = str(_REPO_ROOT)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{sep}{prev}" if prev else root
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "analyze_dashboard failed")
    return json.loads(proc.stdout.strip())


_DOM_ASSESSMENT_READABLE: Dict[str, str] = {
    "wrapper_interstitial": "Wrapper / interstitial — redirect, preview, or gateway-style language or layout.",
    "credential_harvester_risk": "Credential harvester risk — minimal login surface with little real site context.",
    "legitimate_auth_flow_likely": "Legitimate auth flow (likely) — official host with aligned submits and richer chrome.",
    "suspicious_impersonation": "Suspicious impersonation — brand/CTA targets or forms do not align with the page host.",
    "inconclusive": "Inconclusive — DOM anomaly signals are mixed or weak.",
}


def verdict_color(label: str) -> str:
    l = (label or "").lower()
    if "phish" in l or "high" in l:
        return "#ef4444"
    if "uncertain" in l or "suspicious" in l or "inconclusive" in l:
        return "#f59e0b"
    return "#22c55e"


def render_badge(text: str, kind: str = "neutral") -> None:
    colors = {
        "danger": ("#7f1d1d", "#fecaca"),
        "warning": ("#78350f", "#fde68a"),
        "success": ("#14532d", "#bbf7d0"),
        "info": ("#1e3a8a", "#bfdbfe"),
        "neutral": ("#1f2937", "#d1d5db"),
    }
    bg, fg = colors.get(kind, colors["neutral"])
    st.markdown(
        f"<span style='display:inline-block;padding:0.28rem 0.68rem;border-radius:999px;background:{bg};color:{fg};font-size:0.82rem;font-weight:650;margin-right:0.42rem;margin-bottom:0.2rem;'>{text}</span>",
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: Any, help_text: str = "") -> None:
    disp = "N/A" if value is None or value == "" else str(value)
    st.markdown(
        f"""
        <div style="border:1px solid rgba(148,163,184,.38);border-radius:12px;padding:0.95rem 1rem;background:rgba(15,23,42,.42);min-height:108px;display:flex;flex-direction:column;justify-content:space-between;">
          <div style="font-size:.84rem;color:#94a3b8;font-weight:650;margin-bottom:.38rem;">{title}</div>
          <div style="font-size:1.34rem;font-weight:760;color:#f8fafc;line-height:1.2;">{disp}</div>
          <div style="font-size:.79rem;color:#94a3b8;margin-top:.4rem;line-height:1.25;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_layer_section(title: str, description: str) -> None:
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<h3 style='margin-bottom:0.2rem;font-size:1.55rem;'>{title}</h3>", unsafe_allow_html=True)
    st.caption(description)


def _add_status_badges(verdict: Dict[str, Any], gaps: List[str]) -> None:
    conf = str(verdict.get("confidence") or "").lower()
    if conf == "low":
        render_badge("Low confidence", "warning")
    elif conf == "medium":
        render_badge("Medium confidence", "info")
    elif conf == "high":
        render_badge("High confidence", "success")
    label = str(verdict.get("label") or "").lower()
    if "uncertain" in label or "inconclusive" in label:
        render_badge("Inconclusive", "warning")
    if gaps:
        render_badge("Evidence gap", "danger")
    if verdict.get("inactive_site_detected"):
        render_badge("Inactive Site", "warning")
    pctx = str(verdict.get("platform_context_type") or "").strip()
    if pctx and pctx != "unknown":
        render_badge(f"Platform Context: {pctx.replace('_', ' ').title()}", "info")


def render_evidence_summary(
    verdict_reasons: List[str],
    dom_reasons: List[str],
    ai_block: Dict[str, Any],
    gaps: List[str],
    capture_cap: Optional[Dict[str, Any]] = None,
) -> None:
    render_layer_section("Final Explanation / Evidence Summary", "Consolidated explanation across all layers.")
    ai_res = (ai_block or {}).get("ai_result") or {}
    cap = capture_cap or {}
    susp = list(dom_reasons or []) + [f"Verdict reason: {r}" for r in (verdict_reasons or []) if "phish" in r.lower() or "risk" in r.lower()]
    if cap.get("capture_failure_suspicious"):
        susp.append(
            "Live capture failed or was blocked, which can occur with evasive phishing pages."
        )
    legit = [f"AI legit: {x}" for x in (ai_res.get("reasons_for_legitimacy") or [])]
    legit += [f"Verdict reason: {r}" for r in (verdict_reasons or []) if "legit" in r.lower() or "reduce" in r.lower()]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Signals Supporting Phishing**")
        if susp:
            for x in susp[:8]:
                st.markdown(f"- {x}")
        else:
            st.caption("No major signals recorded.")
    with c2:
        st.markdown("**Signals Supporting Legitimacy**")
        if legit:
            for x in legit[:8]:
                st.markdown(f"- {x}")
        else:
            st.caption("No major signals recorded.")
    with c3:
        st.markdown("**Evidence Gaps / Limitations**")
        if gaps:
            for g in gaps:
                st.markdown(f"- {g}")
        else:
            st.caption("No major signals recorded.")


def _show_dashboard(row: Dict[str, Any]) -> None:
    ml = row.get("layer1_ml") or {}
    reinforcement_raw = row.get("reinforcement")
    rein = reinforcement_raw if isinstance(reinforcement_raw, dict) else {}
    htmls = row.get("html_structure") or {}
    hdom = row.get("html_dom_anomaly") or {}
    hpr = row.get("host_path_reasoning") or {}
    html_enrich = row.get("html_dom_enrichment") or {}
    enrichment_signals = row.get("enrichment_signals") or {}
    verdict = row.get("verdict") or {}
    gaps = row.get("evidence_gaps") or []

    # Grand verdict
    st.markdown("---")
    st.markdown("<h2 style='font-size:2rem;margin-bottom:0.15rem;'>Grand Verdict</h2>", unsafe_allow_html=True)
    _add_status_badges(verdict, gaps)
    vlabel = str(verdict.get("label") or "N/A").replace("_", " ").title()
    color = verdict_color(vlabel)
    st.markdown(
        f"<div style='border:1.2px solid {color};border-radius:14px;padding:1.1rem 1.2rem;background:rgba(2,6,23,.55);'>"
        f"<div style='font-size:.9rem;color:#94a3b8;font-weight:650;'>Assessment</div>"
        f"<div style='font-size:1.95rem;font-weight:820;color:{color};margin:.15rem 0 .5rem 0;line-height:1.15;'>{vlabel}</div>"
        f"<div style='color:#dbeafe;font-size:1rem;line-height:1.35;'>This verdict combines URL/host ML scoring, live webpage reinforcement, HTML/DOM structure signals, and optional AI adjudication.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    warn = verdict.get("capture_failure_high_ml_incomplete_warning")
    if warn:
        st.warning(str(warn))
    if verdict.get("inactive_site_detected"):
        st.info(str(verdict.get("inactive_site_explanation") or "Content unavailable - classification limited"))
    cs = verdict.get("combined_score")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Assessment", vlabel, "Final post-policy label.")
    with c2:
        render_metric_card("Confidence", str(verdict.get("confidence") or "N/A").title(), "Interpret with evidence gaps.")
    with c3:
        render_metric_card("Combined Score", f"{float(cs):.3f}" if isinstance(cs, (int, float)) else "N/A", "0 (legit) → 1 (phishing).")
    with c4:
        render_metric_card(
            "Platform Context",
            str(verdict.get("platform_context_type") or "unknown").replace("_", " ").title(),
            str(verdict.get("platform_name") or "No platform match"),
        )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "Original ML Score",
            f"{float(verdict.get('legitimacy_rescue_original_ml_score')):.3f}"
            if isinstance(verdict.get("legitimacy_rescue_original_ml_score"), (int, float))
            else "N/A",
            "Layer-1 score before rescue.",
        )
    with c2:
        render_metric_card(
            "Legitimacy Rescue Applied",
            "Yes" if verdict.get("legitimacy_rescue_applied") else "No",
            "Post-ML downgrade safeguard.",
        )
    with c3:
        render_metric_card(
            "ML Cap After Rescue",
            f"{float(verdict.get('legitimacy_rescue_ml_cap_applied')):.3f}"
            if isinstance(verdict.get("legitimacy_rescue_ml_cap_applied"), (int, float))
            else "N/A",
            "Configured score cap when rescued.",
        )
    with c4:
        render_metric_card(
            "Rescue Target Verdict",
            str(verdict.get("legitimacy_rescue_target_verdict") or "N/A").replace("_", " ").title(),
            "Configured downgrade target.",
        )
    rescue_reasons = verdict.get("legitimacy_rescue_reasons") or []
    rescue_blockers = verdict.get("legitimacy_rescue_blockers") or []
    if rescue_reasons:
        st.markdown("**Legitimacy Rescue Evidence**")
        for r in rescue_reasons:
            st.markdown(f"- {r}")
    if rescue_blockers:
        st.markdown("**Strong Signals Blocking Rescue**")
        for b in rescue_blockers:
            st.markdown(f"- {b}")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card(
            "Strong Legitimacy Override",
            "Triggered" if verdict.get("legitimacy_strong_override_triggered") else "Not triggered",
            "Forces uncertain when strict structural legitimacy conditions all pass.",
        )
    with c2:
        dns_meta = (verdict.get("ml_legitimacy_blend") or {}).get("dns_feature_dampening") or {}
        dns_share = dns_meta.get("dns_contribution_share")
        render_metric_card(
            "DNS Contribution Share",
            f"{float(dns_share):.3f}" if isinstance(dns_share, (int, float)) else "N/A",
            "Approx. share from DNS-related top linear signals.",
        )
    with c3:
        render_metric_card(
            "Hosting Trust Status",
            str(verdict.get("hosting_trust_status") or "hosting_trust_unknown").replace("_", " ").title(),
            "Supporting signal only; never sole verdict driver.",
        )
    override_conds = verdict.get("legitimacy_strong_override_conditions") or []
    if override_conds:
        st.markdown("**Strong Override Conditions Met**")
        for c in override_conds:
            st.markdown(f"- {c}")
    dns_meta = (verdict.get("ml_legitimacy_blend") or {}).get("dns_feature_dampening") or {}
    if dns_meta:
        st.markdown("**DNS Dominance Debug**")
        st.markdown(f"- dns_dominant: {bool(dns_meta.get('dns_dominant'))}")
        st.markdown(f"- dns_dampening_applied: {bool(dns_meta.get('dns_dampening_applied'))}")
        if isinstance(dns_meta.get("dns_dampening_factor"), (int, float)):
            st.markdown(f"- dns_dampening_factor: {float(dns_meta.get('dns_dampening_factor')):.3f}")
        if dns_meta.get("dns_features_detected"):
            st.markdown("- dns_features_detected:")
            for feat in dns_meta.get("dns_features_detected")[:12]:
                st.markdown(f"  - {feat}")
    with st.expander("Rescue Debug", expanded=False):
        dns_meta = (verdict.get("ml_legitimacy_blend") or {}).get("dns_feature_dampening") or {}
        st.write(
            {
                "legitimacy_rescue_applied": verdict.get("legitimacy_rescue_applied"),
                "legitimacy_strong_override_triggered": verdict.get("legitimacy_strong_override_triggered"),
                "legitimacy_strong_override_conditions": verdict.get("legitimacy_strong_override_conditions"),
                "legitimacy_rescue_reasons": verdict.get("legitimacy_rescue_reasons"),
                "legitimacy_rescue_blockers": verdict.get("legitimacy_rescue_blockers"),
                "original_ml_score": verdict.get("legitimacy_rescue_original_ml_score"),
                "effective_ml_score": verdict.get("effective_ml_score"),
                "ml_cap_after_rescue": verdict.get("legitimacy_rescue_ml_cap_applied"),
                "dns_feature_dampening": {
                    "dns_dominant": dns_meta.get("dns_dominant"),
                    "dns_dampening_applied": dns_meta.get("dns_dampening_applied"),
                    "dns_dampening_factor": dns_meta.get("dns_dampening_factor"),
                    "dns_contribution_share": dns_meta.get("dns_contribution_share"),
                    "dns_feature_names": dns_meta.get("dns_features_detected"),
                },
                "hosting_trust_status": verdict.get("hosting_trust_status"),
                "hosting_trust_reasons": verdict.get("hosting_trust_reasons"),
                "hosting_trust_evidence": verdict.get("hosting_trust_evidence"),
                "hosting_trust_mismatches": verdict.get("hosting_trust_mismatches"),
                "cloud_hosted_brand_impersonation": verdict.get("cloud_hosted_brand_impersonation"),
                "untrusted_builder_hosting_signal_applied": verdict.get("untrusted_builder_hosting_signal_applied"),
                "untrusted_builder_hosting_reason": verdict.get("untrusted_builder_hosting_reason"),
                "combined_score_pre_untrusted_builder_downgrade": verdict.get(
                    "combined_score_pre_untrusted_builder_downgrade"
                ),
                "inactive_site_detected": verdict.get("inactive_site_detected"),
                "inactive_site_label": verdict.get("inactive_site_label"),
                "inactive_site_explanation": verdict.get("inactive_site_explanation"),
                "combined_score_pre_inactive_site_overlay": verdict.get("combined_score_pre_inactive_site_overlay"),
                "platform_context_type": verdict.get("platform_context_type"),
                "platform_name": verdict.get("platform_name"),
                "platform_context_reasons": verdict.get("platform_context_reasons"),
                "oauth_providers_detected": verdict.get("oauth_providers_detected"),
                "oauth_brand_mismatch_suppressed": verdict.get("oauth_brand_mismatch_suppressed"),
                "oauth_provider_link_matches": verdict.get("oauth_provider_link_matches"),
                "dormant_phishing_infra_detected": verdict.get("dormant_phishing_infra_detected"),
                "dormant_phishing_infra_reasons": verdict.get("dormant_phishing_infra_reasons"),
                "ml_overconfidence_cap_applied": verdict.get("ml_overconfidence_cap_applied"),
                "ml_overconfidence_cap_reason": verdict.get("ml_overconfidence_cap_reason"),
                "ml_score_before_overconfidence_cap": verdict.get("ml_score_before_overconfidence_cap"),
                "ml_score_after_overconfidence_cap": verdict.get("ml_score_after_overconfidence_cap"),
            }
        )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Layer 1
    render_layer_section(
        "Layer 1: URL / Host ML Triage",
        "Uses trained ML features from the URL and host structure to estimate phishing probability.",
    )
    if ml.get("error"):
        st.warning(f"ML unavailable: {ml.get('error')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Phishing Probability", f"{float(ml.get('phish_proba')):.4f}" if isinstance(ml.get("phish_proba"), (int, float)) else "N/A")
    with c2:
        render_metric_card("Raw Model P(phish)", f"{float(ml.get('phish_proba_model_raw')):.4f}" if isinstance(ml.get("phish_proba_model_raw"), (int, float)) else "N/A")
    with c3:
        render_metric_card("Calibrated P(phish)", f"{float(ml.get('phish_proba_calibrated')):.4f}" if isinstance(ml.get("phish_proba_calibrated"), (int, float)) else "N/A")
    st.caption("What this means: this is the fastest first-pass risk estimate from URL/host features, before deep page reasoning.")
    with st.expander("Layer 1 details / raw features"):
        st.write({"canonical_url": ml.get("canonical_url"), "model_path": ml.get("model_path")})
        tops = ml.get("top_linear_signals") or []
        if tops:
            st.dataframe(tops, use_container_width=True)
        bnotes = ml.get("brand_structure_explanations") or []
        for line in bnotes:
            st.markdown(f"- {line}")

    # Layer 2
    render_layer_section(
        "Layer 2: Live Fetch Reinforcement",
        "Re-fetches the live page in a containerized browser/HTTP fallback to inspect redirects, final URL, blocking, and organization-style behavior.",
    )
    if rein.get("error"):
        st.error(str(rein.get("error")))
    if reinforcement_raw is None:
        st.info("Reinforcement was disabled for this analysis; live capture and click-probe diagnostics are omitted.")
    cap = (rein.get("capture") or {}) if isinstance(rein, dict) else {}
    org = (rein.get("org_style") or {}) if isinstance(rein, dict) else {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Final URL", cap.get("final_url") or "N/A")
    with c2:
        render_metric_card("Redirect Count", cap.get("redirect_count") if cap.get("redirect_count") is not None else "N/A")
    with c3:
        render_metric_card("Cross-domain Redirects", cap.get("cross_domain_redirect_count") if cap.get("cross_domain_redirect_count") is not None else "N/A")
    with c4:
        render_metric_card("Org-style Risk", f"{float(org.get('org_style_risk_score')):.3f}" if isinstance(org.get("org_style_risk_score"), (int, float)) else "N/A")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Input Reg Domain", cap.get("input_registered_domain") or "N/A", "Registered domain from input URL.")
    with c2:
        render_metric_card("Final Reg Domain", cap.get("final_registered_domain") or "N/A", "Registered domain after redirects.")
    with c3:
        render_metric_card("Brand/Domain Mismatch", "Yes" if cap.get("brand_domain_mismatch") else "No")
    with c4:
        render_metric_card("Free Hosting Domain", "Yes" if cap.get("final_domain_is_free_hosting") else "No")
    st.caption("What this means: this layer validates real-world behavior (where the page actually lands and how it behaves).")

    if reinforcement_raw is not None:
        cstat = str(cap.get("capture_status_display") or "unknown").lower()
        suspicious_ctx = bool(cap.get("capture_failure_suspicious"))
        level = str(cap.get("capture_failure_suspicion_level") or "none").lower()
        low_ml_fail = cstat == "failed" and level == "weak"
        if cstat == "success":
            render_badge("Capture Status: Success", "success")
        elif cstat == "partial":
            render_badge("Capture Status: Partial", "warning")
        elif cstat == "failed" and suspicious_ctx and not low_ml_fail:
            render_badge("Capture Status: Failed", "danger")
        elif cstat == "failed":
            render_badge("Capture Status: Failed", "warning")
        else:
            render_badge(f"Capture Status: {cstat.title()}", "neutral")
        if cap.get("capture_failed"):
            render_badge(f"Failure Type: {cap.get('capture_failure_type') or 'n/a'}", "warning")
            render_badge(f"Suspicion Level: {cap.get('capture_failure_suspicion_level') or 'n/a'}", "warning")
        c1, c2 = st.columns(2)
        with c1:
            render_metric_card(
                "Capture Failure Reason",
                cap.get("capture_failure_reason") or ("N/A" if not cap.get("capture_failed") else "—"),
                "Plain-language context only.",
            )
        with c2:
            render_metric_card(
                "Capture Error (raw)",
                (str(cap.get("error"))[:220] if cap.get("error") else "None"),
                "Technical detail from the fetch layer.",
            )
        st.caption(
            "Capture failure is not proof of phishing, but when paired with high ML phishing probability it is treated "
            "as suspicious/evasive context."
        )

    st.markdown("**Strong Signals**")
    strong = enrichment_signals.get("strong_signals") or []
    if strong:
        for s in strong:
            st.markdown(f"- {s}")
    else:
        st.caption("No major signals recorded.")
    st.markdown("**Weak / Contextual Signals**")
    weak = enrichment_signals.get("weak_contextual_signals") or []
    if weak:
        for w in weak:
            st.markdown(f"- {w}")
    else:
        st.caption("No major signals recorded.")
    cp = cap.get("click_probe") or {}
    if reinforcement_raw is None:
        cp = {}
    with st.expander("Click Probe Diagnostics", expanded=False):
        st.markdown(
            "The optional click probe is off by default (`PHISH_ENABLE_CLICK_PROBE=false`). "
            "When enabled, the pipeline may click **one** visible, pattern-matched control (no typing)."
        )
        d1, d2 = st.columns(2)
        with d1:
            render_metric_card("Enabled", "Yes" if cp.get("click_probe_enabled") else "No")
            render_metric_card("Attempted", "Yes" if cp.get("click_probe_attempted") else "No")
            render_metric_card("Candidate Count", cp.get("click_probe_candidate_count", "N/A"))
        with d2:
            render_metric_card("Skip Reason", cp.get("click_probe_skip_reason") or "—")
            render_metric_card("Error", (cp.get("click_probe_error") or "—")[:180])
        sample = cp.get("click_probe_candidate_texts_sample") or []
        if sample:
            st.markdown("**Sample candidate texts**")
            for t in sample[:8]:
                st.markdown(f"- {t}")
        else:
            st.caption("No candidate button/link texts matched the safe probe patterns.")
        render_metric_card("Clicked Text", cp.get("click_probe_text") or "N/A")
        render_metric_card("Domain Changed", cp.get("click_probe_domain_changed", "N/A"))
    for r in org.get("reasons") or []:
        st.markdown(f"- {r}")
    with st.expander("Layer 2 raw capture/org-style"):
        st.write({"capture": cap, "org_style": org})

    # Layer 3
    render_layer_section(
        "Layer 3: HTML / DOM Structure Signals",
        "Examines HTML structure such as forms, password inputs, external form actions, sparse login layouts, support links, and DOM anomalies.",
    )
    hs = htmls.get("html_structure_summary")
    da = hdom.get("html_dom_anomaly_summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Forms", (hs or {}).get("form_count", "N/A"))
    with c2:
        render_metric_card("Password Inputs", (hs or {}).get("password_input_count", "N/A"))
    with c3:
        render_metric_card("External Form Actions", (da or {}).get("form_action_external_domain_count", "N/A"))
    with c4:
        render_metric_card("DOM Anomaly Risk", f"{float(hdom.get('html_dom_anomaly_risk_score')):.3f}" if isinstance(hdom.get("html_dom_anomaly_risk_score"), (int, float)) else "N/A")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Script Tags", html_enrich.get("script_tag_count", "N/A"))
    with c2:
        render_metric_card("Iframes", html_enrich.get("iframe_count", "N/A"))
    with c3:
        render_metric_card("Hidden Inputs", html_enrich.get("hidden_input_count", "N/A"))
    with c4:
        render_metric_card("External Script Domains", html_enrich.get("external_script_domain_count", "N/A"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("External Form Actions", html_enrich.get("external_form_action_count", "N/A"))
    with c2:
        render_metric_card("Password + External Action", "Yes" if html_enrich.get("password_input_external_action") else "No")
    with c3:
        render_metric_card("Sparse Login-like Layout", "Yes" if html_enrich.get("sparse_login_like_layout") else "No")
    with c4:
        render_metric_card("Missing Org Elements", "Yes" if html_enrich.get("missing_org_elements") else "No")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Detected Language", html_enrich.get("detected_language") or "N/A")
    with c2:
        conf = html_enrich.get("detected_language_confidence")
        render_metric_card("Language Confidence", f"{float(conf):.3f}" if isinstance(conf, (int, float)) else "N/A")
    with c3:
        render_metric_card("Language Detection", "Available" if html_enrich.get("language_detection_available") else "N/A")
    with c4:
        render_metric_card("Language Mismatch (Contextual)", "Yes" if html_enrich.get("language_mismatch_contextual_signal") else "No")
    if html_enrich.get("language_detection_error"):
        st.caption(f"Language detection note: {html_enrich.get('language_detection_error')}")
    st.caption(
        "Detected page language may help identify suspicious localization mismatches, but multilingual legitimate sites "
        "are common, so this is treated as contextual evidence only."
    )
    st.caption("What this means: this layer checks trust-action consistency (labels, links, forms, and page behavior patterns).")
    assess_key = str(hdom.get("html_dom_visual_assessment") or "inconclusive")
    st.markdown(f"**DOM pattern:** {_DOM_ASSESSMENT_READABLE.get(assess_key, _DOM_ASSESSMENT_READABLE['inconclusive'])}")
    for r in hdom.get("html_dom_anomaly_reasons") or []:
        st.markdown(f"- {r}")
    if hpr:
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Host Class", str(hpr.get("host_identity_class") or "N/A").replace("_", " ").title())
        with c2:
            render_metric_card("Host Legitimacy", str(hpr.get("host_legitimacy_confidence") or "N/A").title())
        with c3:
            render_metric_card("Path Fit", str(hpr.get("path_fit_assessment") or "N/A").replace("_", " ").title())
    with st.expander("Layer 3 raw HTML/DOM/host-path evidence"):
        st.write({"html_structure": htmls, "html_dom_anomaly": hdom, "html_dom_enrichment": html_enrich, "host_path_reasoning": hpr})

    # Layer 4
    render_layer_section(
        "Layer 4: AI Adjudication",
        "Uses AI adjudication only as an explainability/review layer, especially when model and rule-based signals disagree or confidence is low.",
    )
    ai = verdict.get("ai_adjudication") or {}
    if not ai.get("enabled", True):
        st.info("AI adjudication disabled.")
    elif not ai.get("ran"):
        msg = ai.get("skip_reason") or "not_run"
        if ai.get("ai_error"):
            msg += f" ({ai.get('ai_error')})"
        st.info(f"AI adjudication did not run: {msg}")
        if ai.get("eligibility_reasons"):
            st.caption("Eligibility: " + ", ".join(ai.get("eligibility_reasons") or []))
    else:
        ai_res = ai.get("ai_result") or {}
        adj = ai.get("adjustment") or {}
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("AI Assessment", str(ai_res.get("ai_assessment") or "N/A").replace("_", " ").title())
        with c2:
            render_metric_card("AI Confidence", str(ai_res.get("ai_confidence") or "N/A").title())
        with c3:
            render_metric_card("AI Adjustment", f"{float(adj.get('adjustment_applied') or 0):+.3f}")
        st.caption("What this means: AI can explain and nudge scores within strict bounds; it is not the sole source of truth.")
        st.markdown(str(ai_res.get("summary") or "No AI summary provided."))
        for r in ai_res.get("reasons_for_legitimacy") or []:
            st.markdown(f"- Legit signal: {r}")
        for r in ai_res.get("reasons_for_suspicion") or []:
            st.markdown(f"- Suspicion signal: {r}")
        for r in ai_res.get("uncertainty_notes") or []:
            st.markdown(f"- Uncertainty: {r}")
    with st.expander("Layer 4 raw AI adjudication JSON"):
        st.write(ai)

    # Final explanation
    render_evidence_summary(
        verdict_reasons=verdict.get("reasons") or [],
        dom_reasons=hdom.get("html_dom_anomaly_reasons") or [],
        ai_block=ai,
        gaps=gaps,
        capture_cap=cap,
    )

    # Evidence gaps kept visible
    st.markdown("---")
    st.subheader("Evidence Gaps")
    if gaps:
        for g in gaps:
            st.warning(g)
    else:
        st.success("No major evidence gaps recorded.")

    with st.expander("Raw JSON (all output fields)"):
        st.code(json.dumps(row, ensure_ascii=False, indent=2), language="json")

    with st.expander("Known Limitations"):
        st.markdown("- JavaScript-heavy pages may require stabilization time before reliable scraping.")
        st.markdown("- Rapid page swaps can evade screenshot or DOM capture.")
        st.markdown("- PhishStats provides phishing URLs only, so legitimate baselines must come from curated brand references.")
        st.markdown("- AI adjudication is used for explanation and review, not as the sole source of truth.")


def main() -> None:
    st.set_page_config(page_title="Phishing analysis", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1200px;}
        div[data-testid="stMetricValue"] {font-size: 1.25rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Phishing analysis dashboard")
    st.caption(
        "Layer-1 ML triage + optional containerized live checks. "
        "Screenshot comparison UI is deprecated — see `archive/legacy/frontend_screenshot_legacy.py`."
    )
    with st.expander("How to read this dashboard", expanded=True):
        st.markdown("1. Start with **Grand Verdict**.")
        st.markdown("2. Review each layer for supporting evidence.")
        st.markdown("3. Use **Evidence Summary** and **Known Limitations** for interpretation.")

    url = st.text_input("URL to analyze", placeholder="https://example.com")
    reinforcement = st.checkbox("Run reinforcement (Playwright / HTTP in container)", value=True)
    layer1_dns = st.checkbox("Layer-1 DNS lookups (slower)", value=False)
    ai_adjudication = st.checkbox("Enable AI adjudication (borderline/disagreement only)", value=True)

    if st.button("Analyze", type="primary"):
        url = (url or "").strip()
        if not url:
            st.warning("Enter a URL.")
            return
        with st.spinner("Analyzing…"):
            try:
                row = _run_dashboard_subprocess(url, reinforcement, layer1_dns, ai_adjudication)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed: {exc}")
                logger.error(traceback.format_exc())
                return
        _show_dashboard(row)


if __name__ == "__main__":
    main()
