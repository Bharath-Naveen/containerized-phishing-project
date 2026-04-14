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
from typing import Any, Dict, Optional

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


def _show_dashboard(row: Dict[str, Any]) -> None:
    ml = row.get("layer1_ml") or {}
    rein = row.get("reinforcement") or {}
    htmls = row.get("html_structure") or {}
    hdom = row.get("html_dom_anomaly") or {}
    hpr = row.get("host_path_reasoning") or {}
    verdict = row.get("verdict") or {}
    gaps = row.get("evidence_gaps") or []

    st.subheader("Verdict")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Assessment", verdict.get("label", "—").replace("_", " ").title())
    with c2:
        st.metric("Confidence", str(verdict.get("confidence", "—")).title())
    with c3:
        cs = verdict.get("combined_score")
        st.metric("Combined score", f"{cs:.3f}" if isinstance(cs, (int, float)) else "—")
    for r in verdict.get("reasons") or []:
        st.markdown(f"- {r}")

    st.subheader("Layer-1 ML (URL / host)")
    if ml.get("error"):
        st.warning(f"ML: {ml['error']}")
    else:
        st.metric("Phishing probability", f"{float(ml.get('phish_proba') or 0):.4f}")
        pr = ml.get("phish_proba_model_raw")
        pc = ml.get("phish_proba_calibrated")
        if pr is not None and pc is not None and abs(float(pr) - float(pc)) > 1e-5:
            st.caption(f"Raw model P(phish) {float(pr):.4f} → calibrated {float(pc):.4f}")
        st.caption(f"Model: `{ml.get('model_path', '')}`")
        st.caption(f"Canonical URL: `{ml.get('canonical_url', '')}`")
        tops = ml.get("top_linear_signals") or []
        if tops:
            with st.expander("Top linear model signals (logistic regression only)"):
                st.dataframe(tops, use_container_width=True)
        bnotes = ml.get("brand_structure_explanations") or []
        if bnotes:
            with st.expander("Brand / host structure (Layer-1 explanations)"):
                for line in bnotes:
                    st.markdown(f"- {line}")

    st.subheader("Reinforcement (live fetch / org-style)")
    if rein.get("error"):
        st.error(rein["error"])
    elif not rein:
        st.info("Reinforcement not run.")
    else:
        cap = rein.get("capture") or {}
        st.write(
            {
                "final_url": cap.get("final_url"),
                "redirect_count": cap.get("redirect_count"),
                "cross_domain_redirect_count": cap.get("cross_domain_redirect_count"),
                "capture_strategy": cap.get("capture_strategy"),
                "capture_blocked": cap.get("capture_blocked"),
                "error": cap.get("error"),
            }
        )
        org = rein.get("org_style") or {}
        st.metric("Org-style risk", f"{float(org.get('org_style_risk_score') or 0):.3f}")
        for r in org.get("reasons") or []:
            st.markdown(f"- {r}")

    st.subheader("Evidence gaps")
    if gaps:
        for g in gaps:
            st.warning(g)
    else:
        st.success("No major evidence gaps recorded.")

    with st.expander("Raw JSON"):
        st.code(json.dumps(row, ensure_ascii=False, indent=2), language="json")

    st.subheader("HTML Structure Signals")
    hs = htmls.get("html_structure_summary")
    if not hs:
        st.info(f"HTML structure unavailable ({htmls.get('html_structure_error') or 'not_available'}).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Forms", int(hs.get("form_count") or 0))
        with c2:
            st.metric("Password inputs", int(hs.get("password_input_count") or 0))
        with c3:
            st.metric("Cross-domain form action", "Yes" if hs.get("cross_domain_form_action") else "No")
        with c4:
            score = htmls.get("html_structure_risk_score")
            st.metric("HTML structure risk", f"{float(score):.3f}" if isinstance(score, (int, float)) else "—")
        for r in htmls.get("html_structure_reasons") or []:
            st.markdown(f"- {r}")
        st.write(
            {
                "title": hs.get("title"),
                "suspicious_phrase_hits": hs.get("suspicious_phrase_hits"),
                "brand_terms_found_in_text": hs.get("brand_terms_found_in_text"),
                "button_texts": hs.get("button_texts"),
                "nav_link_count": hs.get("nav_link_count"),
                "footer_link_count": hs.get("footer_link_count"),
                "sparse_login_like_layout": hs.get("sparse_login_like_layout"),
                "brand_mismatch_from_html_context": hs.get("brand_mismatch_from_html_context"),
                "has_forgot_password_link": hs.get("has_forgot_password_link"),
                "has_support_help_links": hs.get("has_support_help_links"),
            }
        )

    st.subheader("HTML / DOM anomaly review")
    da = hdom.get("html_dom_anomaly_summary")
    if not da:
        st.info(f"DOM anomaly layer unavailable ({hdom.get('html_dom_anomaly_error') or 'not_available'}).")
    else:
        assess_key = str(hdom.get("html_dom_visual_assessment") or "inconclusive")
        st.caption(_DOM_ASSESSMENT_READABLE.get(assess_key, _DOM_ASSESSMENT_READABLE["inconclusive"]))
        c1, c2, c3 = st.columns(3)
        with c1:
            rs = hdom.get("html_dom_anomaly_risk_score")
            st.metric("DOM anomaly risk", f"{float(rs):.3f}" if isinstance(rs, (int, float)) else "—")
        with c2:
            st.metric("Pattern label", assess_key.replace("_", " ").title())
        with c3:
            st.metric("External form actions", int(da.get("form_action_external_domain_count") or 0))
        st.markdown("**Why (interpretable)**")
        for r in hdom.get("html_dom_anomaly_reasons") or []:
            st.markdown(f"- {r}")
        with st.expander("Mismatch evidence (compact)"):
            st.markdown("**Page context (heuristic)**")
            st.write(
                {
                    "page_family": da.get("page_family"),
                    "trust_action_context": da.get("trust_action_context"),
                    "strong_impersonation_context": da.get("strong_impersonation_context"),
                    "content_rich_profile": da.get("content_rich_profile"),
                    "dampener_factor": da.get("content_rich_dampener_factor"),
                    "strict_anchor_filter": da.get("strict_anchor_filter_active"),
                }
            )
            st.markdown("**Anchor / CTA vs target**")
            for x in da.get("suspicious_anchor_target_mismatches") or []:
                st.markdown(f"- `{x}`")
            pairs = da.get("top_anchor_texts_with_target_domains") or []
            if pairs:
                st.dataframe(pairs, use_container_width=True)
            st.markdown("**Forms**")
            st.write(
                {
                    "external_form_action_count": da.get("form_action_external_domain_count"),
                    "js_submit_only_login_pattern": da.get("js_submit_only_login_pattern"),
                    "suspicious_credential_collection": da.get("suspicious_credential_collection_pattern"),
                }
            )
            st.markdown("**Brand / resource alignment**")
            st.write(
                {
                    "title_brand_domain_mismatch": da.get("title_brand_domain_mismatch"),
                    "body_brand_domain_mismatch": da.get("body_brand_domain_mismatch"),
                    "logo_domain_mismatch": da.get("logo_domain_mismatch"),
                    "resource_domains_top": (da.get("resource_domains_summary") or [])[:8],
                }
            )
            st.markdown("**Interstitial / wrapper**")
            st.write(
                {
                    "interstitial_or_preview_pattern": da.get("interstitial_or_preview_pattern"),
                    "continue_to_destination_phrase": da.get("continue_to_destination_phrase_present"),
                    "destination_preview_phrase": da.get("destination_preview_phrase_present"),
                    "redirect_countdown_phrase": da.get("redirect_countdown_phrase_present"),
                    "wrapper_page_pattern": da.get("wrapper_page_pattern"),
                }
            )
            st.markdown("**Structure**")
            st.write(
                {
                    "login_harvester_pattern": da.get("login_harvester_pattern"),
                    "suspicious_minimal_login_clone": da.get("suspicious_minimal_login_clone_pattern"),
                    "missing_ecosystem_context": da.get("missing_real_ecosystem_context"),
                    "strong_branding_non_official_host": da.get("strong_branding_without_official_domain"),
                }
            )

    st.subheader("AI Adjudication")
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
            st.metric("AI assessment", str(ai_res.get("ai_assessment", "—")).replace("_", " "))
        with c2:
            st.metric("AI confidence", str(ai_res.get("ai_confidence", "—")))
        with c3:
            st.metric("AI adjustment", f"{float(adj.get('adjustment_applied') or 0):+.3f}")
        st.caption(str(ai_res.get("summary") or "").strip() or "No AI summary provided.")
        for r in ai_res.get("reasons_for_legitimacy") or []:
            st.markdown(f"- Legit signal: {r}")
        for r in ai_res.get("reasons_for_suspicion") or []:
            st.markdown(f"- Suspicion signal: {r}")
        for r in ai_res.get("uncertainty_notes") or []:
            st.markdown(f"- Uncertainty: {r}")

    st.subheader("Host / Path reasoning")
    if not hpr:
        st.info(f"Host/path reasoning unavailable ({row.get('host_path_reasoning_error') or 'not_available'}).")
    else:
        decomp = hpr.get("url_decomposition") or {}
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Host class", str(hpr.get("host_identity_class") or "—").replace("_", " "))
        with c2:
            st.metric("Host legitimacy", str(hpr.get("host_legitimacy_confidence") or "—").title())
        with c3:
            st.metric("Path fit", str(hpr.get("path_fit_assessment") or "—").replace("_", " "))
        for r in hpr.get("host_legitimacy_reasons") or []:
            st.markdown(f"- Host reason: {r}")
        for r in hpr.get("path_fit_reasons") or []:
            st.markdown(f"- Path reason: {r}")
        with st.expander("Host/path decomposition"):
            st.write(decomp)


def main() -> None:
    st.set_page_config(page_title="Phishing analysis", layout="wide")
    st.title("Phishing analysis dashboard")
    st.caption(
        "Layer-1 ML triage + optional containerized live checks. "
        "Screenshot comparison UI is deprecated — see `archive/legacy/frontend_screenshot_legacy.py`."
    )

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
