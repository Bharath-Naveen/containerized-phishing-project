"""Layer-1 ML + optional reinforcement capture → dashboard JSON (no screenshot-vs-reference flow).

The optional **official-brand apex cap** below is a *temporary UX safety net* only. Primary trust
should be the retrained Layer-1 model and structural brand features in
:mod:`src.pipeline.features.brand_signals`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

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
from .verdict_policy import Verdict3WayConfig, verdict_3way

logger = logging.getLogger(__name__)

# Fallback only: if raw model score is still high on a host under a known official registrable family.
_OFFICIAL_BRAND_APEX_PHISH_CAP = 0.32


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
    host_path_reasoning: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    html_dom_risk: Optional[float],
    legitimacy_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Bounded combined-score rescue when multiple legitimacy layers align."""
    out = dict(verdict)
    c0 = out.get("combined_score")
    if not isinstance(c0, (int, float)):
        out["legitimacy_rescue_applied"] = False
        return out
    hp = host_path_reasoning or {}
    dom = html_dom_summary or {}
    conf_high = str(hp.get("host_legitimacy_confidence") or "") == "high"
    path_plaus = str(hp.get("path_fit_assessment") or "") == "plausible"
    fam = str(dom.get("page_family") or "")
    content_family = fam in {"public_content_platform", "content_feed_forum_aggregator", "article_news", "public_docs_or_reference"}
    dom_low = (float(html_dom_risk) if isinstance(html_dom_risk, (int, float)) else 0.0) <= 0.26
    blocked = any(
        (
            str(hp.get("host_identity_class") or "") == "suspicious_host_pattern",
            bool(dom.get("trust_action_context")),
            bool(dom.get("suspicious_credential_collection_pattern")),
            int(dom.get("form_action_external_domain_count") or 0) > 0,
            bool(dom.get("login_harvester_pattern")),
            bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern")),
            bool(dom.get("trust_surface_brand_domain_mismatch")),
            int(dom.get("anchor_strong_mismatch_count") or 0) > 0,
            bool(legitimacy_bundle.get("suspicious_form_action_cross_origin")),
            not bool(legitimacy_bundle.get("no_deceptive_token_placement", True)),
            not bool(legitimacy_bundle.get("no_free_hosting_signal", True)),
        )
    )
    if blocked or not (conf_high and path_plaus and content_family and dom_low):
        out["legitimacy_rescue_applied"] = False
        return out
    rescue = 0.0
    if c0 >= 0.55:
        rescue = 0.12
    elif c0 >= 0.48:
        rescue = 0.08
    elif c0 >= 0.40:
        rescue = 0.05
    c1 = max(0.0, float(c0) - rescue)
    out["combined_score_pre_legitimacy_rescue"] = float(c0)
    out["combined_score"] = c1
    out["legitimacy_rescue_applied"] = rescue > 0
    out["legitimacy_rescue_adjustment"] = -rescue
    if rescue > 0:
        out["reasons"] = list(out.get("reasons") or []) + [
            f"Bounded legitimacy rescue applied ({-rescue:+.3f}) from aligned host/path/content evidence with low DOM anomaly."
        ]
    return out


def no_phishing_evidence_guard(
    *,
    html_structure_summary: Optional[Dict[str, Any]],
    html_dom_summary: Optional[Dict[str, Any]],
    html_dom_risk: Optional[float],
    host_path_reasoning: Optional[Dict[str, Any]],
) -> bool:
    """Hard guard: if all major phishing indicators are absent, force legitimate."""
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
    url = (url or "").strip()
    evidence_gaps: List[str] = []
    ml = predict_layer1(url, use_dns=layer1_use_dns)
    ml = _apply_official_brand_apex_cap(ml, url)
    if ml.get("error"):
        evidence_gaps.append("Layer-1 ML did not produce a score (missing model or feature error).")

    reinforcement_block: Optional[Dict[str, Any]] = None
    org_risk_raw = 0.0
    capture_json: Optional[Dict[str, Any]] = None
    if reinforcement:
        cfg = PipelineConfig.from_env()
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
                    "redirect_count": cap.redirect_count,
                    "cross_domain_redirect_count": cap.cross_domain_redirect_count,
                    "settled_successfully": cap.settled_successfully,
                    "html_path": cap.html_path,
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

    verdict = _verdict_from_scores(
        ml_eff,
        float(p_raw) if p_raw is not None else None,
        float(p_cal) if p_cal is not None else None,
        org_risk_raw,
        org_adj,
        verdict_cfg=verdict_cfg,
    )
    verdict = _apply_legitimacy_rescue_on_verdict(
        verdict,
        host_path_reasoning=host_path_reasoning,
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        html_dom_risk=html_dom.get("html_dom_anomaly_risk_score"),
        legitimacy_bundle=bundle,
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
    verdict["html_structure_risk_score"] = html_structure.get("html_structure_risk_score")
    verdict["html_structure_reasons"] = html_structure.get("html_structure_reasons")
    verdict["html_dom_anomaly_risk_score"] = html_dom.get("html_dom_anomaly_risk_score")
    verdict["html_dom_anomaly_reasons"] = html_dom.get("html_dom_anomaly_reasons")
    verdict["html_dom_visual_assessment"] = html_dom.get("html_dom_visual_assessment")
    verdict["host_path_identity_class"] = (host_path_reasoning or {}).get("host_identity_class")
    verdict["host_path_legitimacy_confidence"] = (host_path_reasoning or {}).get("host_legitimacy_confidence")
    verdict["host_path_fit_assessment"] = (host_path_reasoning or {}).get("path_fit_assessment")
    guard_on = no_phishing_evidence_guard(
        html_structure_summary=html_structure.get("html_structure_summary"),
        html_dom_summary=html_dom.get("html_dom_anomaly_summary"),
        html_dom_risk=html_dom.get("html_dom_anomaly_risk_score"),
        host_path_reasoning=host_path_reasoning,
    )
    verdict = _apply_no_phishing_evidence_override(verdict, guard_triggered=guard_on, verdict_cfg=verdict_cfg)

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

    out: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "input_url": url,
        "layer1_ml": ml,
        "reinforcement": reinforcement_block,
        "html_structure": html_structure,
        "html_dom_anomaly": html_dom,
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
