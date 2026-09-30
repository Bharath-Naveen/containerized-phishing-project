"""Optional bounded OpenAI adjudication for borderline/disagreement cases.

This layer nudges the combined score; it never replaces ML+reinforcement.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .verdict_policy import Verdict3WayConfig, verdict_3way


@dataclass(frozen=True)
class AIAdjudicationConfig:
    max_adjustment: float = 0.12
    narrow_legit_down_max_adjustment: float = 0.18
    high_legit_content_down_max_adjustment: float = 0.22
    strong_high_guard: float = 0.70
    strong_low_guard: float = 0.15
    disagreement_gap: float = 0.45
    base_model: str = "gpt-4o-mini"


def _trust_anchor_present(bundle: Dict[str, Any]) -> bool:
    return bool(
        bundle.get("official_registrable_anchor")
        or bundle.get("official_domain_family")
        or bundle.get("strong_trust_anchor")
    )


def _html_missing_for_ml_capture_miss_review(
    *,
    html_capture_missing_reason: Optional[str],
    html_structure_error: Optional[str],
    html_dom_anomaly_error: Optional[str],
) -> bool:
    if html_capture_missing_reason == "html_not_available":
        return True
    if html_structure_error in ("missing_html_path", "html_path_not_found"):
        return True
    if html_dom_anomaly_error == "missing_html_path":
        return True
    return False


def _auth_like_url(url: str) -> bool:
    s = (url or "").lower()
    return any(
        t in s
        for t in (
            "login",
            "signin",
            "verify",
            "account",
            "recovery",
            "checkout",
            "billing",
            "dashboard",
            "admin",
            "portal",
        )
    )


def should_run_ai_adjudication(
    *,
    pre_ai_combined: Optional[float],
    ml_effective_score: Optional[float],
    org_risk_adjusted: float,
    bundle: Dict[str, Any],
    pre_verdict: str,
    input_url: str,
    html_dom_anomaly_summary: Optional[Dict[str, Any]] = None,
    host_path_reasoning: Optional[Dict[str, Any]] = None,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
    force_ml_phishing_capture_miss_review: bool = False,
) -> Tuple[bool, List[str]]:
    cfg = verdict_cfg or Verdict3WayConfig()
    reasons: List[str] = []
    if pre_ai_combined is None:
        return False, ["no_combined_score"]
    if force_ml_phishing_capture_miss_review:
        reasons.append("ml_predicted_phishing_but_pre_verdict_legitimate")

    c = float(pre_ai_combined)
    if cfg.combined_low < c < cfg.combined_high:
        reasons.append("pre_ai_uncertain_band")
    if 0.30 <= c <= 0.70:
        reasons.append("pre_ai_borderline_band")

    if ml_effective_score is not None:
        ml = float(ml_effective_score)
        org = float(org_risk_adjusted)
        if abs(ml - org) >= 0.45 and ((ml >= 0.65 and org <= 0.20) or (ml <= 0.20 and org >= 0.65)):
            reasons.append("ml_reinforcement_material_disagreement")

    tricky_legit_candidate = bool(
        (bundle.get("official_registrable_anchor") or bundle.get("official_domain_family"))
        and _auth_like_url(input_url)
        and bundle.get("no_brand_mismatch")
        and bundle.get("no_free_hosting_signal")
        and bundle.get("no_deceptive_token_placement")
        and not bundle.get("suspicious_form_action_cross_origin")
    )
    if tricky_legit_candidate and pre_verdict in {"uncertain", "likely_phishing"}:
        reasons.append("tricky_legit_candidate")

    dom = html_dom_anomaly_summary or {}
    hp = host_path_reasoning or {}
    family = str(dom.get("page_family") or "")
    content_family = family in {"content_feed_forum_aggregator", "article_news", "public_docs_or_reference"}
    no_cred_capture = not any(
        (
            bool(dom.get("trust_action_context")),
            bool(dom.get("suspicious_credential_collection_pattern")),
            int(dom.get("form_action_external_domain_count") or 0) > 0,
            bool(dom.get("login_harvester_pattern")),
            bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern")),
            bool(dom.get("trust_surface_brand_domain_mismatch")),
            int(dom.get("anchor_strong_mismatch_count") or 0) > 0,
        )
    )
    high_legit_host = str(hp.get("host_legitimacy_confidence") or "") == "high"
    over_suspicious_ml = c >= max(cfg.combined_high, 0.48) or (ml_effective_score is not None and float(ml_effective_score) >= 0.58)
    if content_family and high_legit_host and no_cred_capture and over_suspicious_ml:
        reasons.append("high_legit_content_over_suspicious_ml")

    return (len(reasons) > 0), reasons


def _build_evidence_packet(
    *,
    input_url: str,
    layer1_ml: Dict[str, Any],
    reinforcement: Optional[Dict[str, Any]],
    verdict_pre_ai: Dict[str, Any],
    legitimacy_bundle: Dict[str, Any],
    html_structure_summary: Optional[Dict[str, Any]],
    html_structure_risk_score: Optional[float],
    html_structure_reasons: Optional[List[str]],
    html_dom_anomaly_summary: Optional[Dict[str, Any]],
    html_dom_anomaly_risk_score: Optional[float],
    html_dom_anomaly_reasons: Optional[List[str]],
    html_dom_visual_assessment: Optional[str],
    host_path_reasoning: Optional[Dict[str, Any]],
    evidence_gaps: List[str],
    capture_json: Optional[Dict[str, Any]],
    html_capture_missing_reason: Optional[str] = None,
    html_structure_error: Optional[str] = None,
    html_dom_anomaly_error: Optional[str] = None,
    ml_phishing_capture_miss_review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cap = (reinforcement or {}).get("capture") or {}
    org = (reinforcement or {}).get("org_style") or {}
    top = layer1_ml.get("top_linear_signals") or []
    top = top[:8]
    packet: Dict[str, Any] = {
        "input_url": input_url,
        "canonical_url": layer1_ml.get("canonical_url"),
        "final_url": cap.get("final_url"),
        "layer1_ml": {
            "phish_proba_raw": layer1_ml.get("phish_proba_model_raw"),
            "phish_proba_calibrated": layer1_ml.get("phish_proba_calibrated"),
            "phish_proba_post_cap": layer1_ml.get("phish_proba"),
            "predicted_phishing": layer1_ml.get("predicted_phishing"),
            "top_linear_signals": top,
        },
        "reinforcement": {
            "org_style_risk_raw": verdict_pre_ai.get("org_risk_raw"),
            "org_style_risk_adjusted": verdict_pre_ai.get("org_risk_adjusted"),
            "free_hosting_match": org.get("free_hosting_match"),
            "brand_domain_mismatches": org.get("brand_domain_mismatches"),
            "capture_error": cap.get("error"),
            "capture_blocked": cap.get("capture_blocked"),
            "capture_strategy": cap.get("capture_strategy"),
            "redirect_count": cap.get("redirect_count"),
            "cross_domain_redirect_count": cap.get("cross_domain_redirect_count"),
        },
        "legitimacy_bundle": legitimacy_bundle,
        "html_structure": {
            "summary": html_structure_summary,
            "risk_score": html_structure_risk_score,
            "reasons": (html_structure_reasons or [])[:6],
        },
        "html_dom_anomaly": {
            "summary": html_dom_anomaly_summary,
            "risk_score": html_dom_anomaly_risk_score,
            "reasons": (html_dom_anomaly_reasons or [])[:8],
            "visual_assessment": html_dom_visual_assessment,
        },
        "host_path_reasoning": host_path_reasoning,
        "verdict_before_ai": {
            "label": verdict_pre_ai.get("label"),
            "combined_score": verdict_pre_ai.get("combined_score"),
            "combined_without_reinforcement": verdict_pre_ai.get("combined_score_without_reinforcement"),
            "reasons": (verdict_pre_ai.get("reasons") or [])[:6],
        },
        "evidence_gaps": evidence_gaps[:6],
    }
    if capture_json:
        packet["capture_hints"] = {
            "title": str(capture_json.get("title") or "")[:240],
            "visible_text_snippet": str(capture_json.get("visible_text") or "")[:700],
        }
    packet["ml_phishing_capture_miss_review"] = ml_phishing_capture_miss_review or {
        "predicted_phishing": bool(layer1_ml.get("predicted_phishing")),
        "capture_failed": cap.get("capture_failed"),
        "capture_failure_type": cap.get("capture_failure_type"),
        "html_capture_missing_reason": html_capture_missing_reason,
        "html_structure_error": html_structure_error,
        "html_dom_anomaly_error": html_dom_anomaly_error,
        "official_registrable_anchor": legitimacy_bundle.get("official_registrable_anchor"),
        "official_domain_family": legitimacy_bundle.get("official_domain_family"),
        "strong_trust_anchor": legitimacy_bundle.get("strong_trust_anchor"),
        "block_ai_legitimizing_adjustment": False,
    }
    return packet


def _parse_json_payload(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(raw[s : e + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def call_ai_adjudicator(evidence_packet: Dict[str, Any], *, cfg: Optional[AIAdjudicationConfig] = None) -> Dict[str, Any]:
    c = cfg or AIAdjudicationConfig()
    if not os.getenv("OPENAI_API_KEY"):
        return {"ran": False, "ai_error": "missing_OPENAI_API_KEY"}
    try:
        from openai import OpenAI
    except Exception as e:  # noqa: BLE001
        return {"ran": False, "ai_error": f"openai_import_error:{e}"}

    model = os.getenv("PHISH_AI_ADJUDICATION_MODEL", c.base_model)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = (
        "You are a cautious phishing adjudicator.\n"
        "Use ONLY supplied structured evidence. Do not invent facts.\n"
        "Do not trust familiar brands by name alone.\n"
        "Do not mark as phishing by brand mention alone.\n"
        "If evidence gaps exist, include uncertainty notes.\n"
        "html_dom_anomaly is a compact summary of link/form/resource alignment vs the page host and brand cues (no raw HTML).\n"
        "It includes page_family (e.g. article/news, feed/forum, auth, checkout, wrapper) and trust_action_context.\n"
        "Treat many outbound links, third-party assets, and incidental brand mentions in article/feed bodies as NORMAL for content platforms unless there is phishing-specific evidence: credential capture (password/email, cross-host form posts), wrapper/interstitial takeover language, misleading trust-action anchor targets, or login-harvest layout signals.\n"
        "Do not treat content-platform DOM noise alone as strong impersonation.\n"
        "host_path_reasoning provides host identity class, legitimacy confidence, and path_fit_assessment; reason from host identity plus path conformity, not brand familiarity.\n"
        "If there is no credential capture, no trust-action behavior, and no impersonation evidence, do NOT rely on ML score alone; treat ML as potentially wrong.\n"
        "Output STRICT JSON with keys:\n"
        "ai_assessment (likely_legitimate|uncertain|likely_phishing),\n"
        "ai_confidence (low|medium|high),\n"
        "adjustment_direction (down|none|up),\n"
        "adjustment_magnitude (0.0..0.15),\n"
        "summary (string),\n"
        "reasons_for_legitimacy (array of strings),\n"
        "reasons_for_suspicion (array of strings),\n"
        "uncertainty_notes (array of strings)."
    )
    review = (
        (evidence_packet.get("ml_phishing_capture_miss_review") or {})
        if isinstance(evidence_packet, dict)
        else {}
    )
    if review.get("block_ai_legitimizing_adjustment"):
        system_prompt += (
            "\nHARD CONSTRAINT (ml_phishing_capture_miss_review): Layer-1 predicted phishing, live capture failed, "
            "and HTML/DOM validation is missing; no trusted-domain anchor is asserted in the evidence. "
            "Do NOT output ai_assessment likely_legitimate. Do NOT use adjustment_direction down (toward legitimacy). "
            "Prefer uncertain with adjustment_direction none or up within magnitude limits."
        )
    user_prompt = "Evidence JSON:\n" + json.dumps(evidence_packet, ensure_ascii=False)
    r = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    text = (r.choices[0].message.content or "").strip()
    parsed = _parse_json_payload(text)
    out = {
        "ran": True,
        "model": model,
        "raw_text": text,
        "ai_assessment": str(parsed.get("ai_assessment") or "uncertain"),
        "ai_confidence": str(parsed.get("ai_confidence") or "low"),
        "adjustment_direction": str(parsed.get("adjustment_direction") or "none"),
        "adjustment_magnitude": float(parsed.get("adjustment_magnitude") or 0.0),
        "summary": str(parsed.get("summary") or ""),
        "reasons_for_legitimacy": [str(x) for x in (parsed.get("reasons_for_legitimacy") or [])][:6],
        "reasons_for_suspicion": [str(x) for x in (parsed.get("reasons_for_suspicion") or [])][:6],
        "uncertainty_notes": [str(x) for x in (parsed.get("uncertainty_notes") or [])][:6],
    }
    return out


def _bounded_magnitude(pre_ai_score: float, requested: float, cfg: AIAdjudicationConfig) -> float:
    req = max(0.0, min(cfg.max_adjustment, float(requested)))
    center_weight = max(0.20, 1.0 - min(1.0, abs(pre_ai_score - 0.5) / 0.35))
    return req * center_weight


def _is_borderline_legit_candidate(
    *,
    pre_ai_score: float,
    pre_ai_verdict: str,
    ai_result: Dict[str, Any],
    adjudication_context: Optional[Dict[str, Any]],
) -> bool:
    if not (0.44 <= pre_ai_score <= 0.54):
        return False
    if pre_ai_verdict != "uncertain":
        return False
    if str(ai_result.get("ai_assessment") or "") != "likely_legitimate":
        return False
    if str(ai_result.get("ai_confidence") or "") not in {"medium", "high"}:
        return False
    ctx = adjudication_context or {}
    bundle = (ctx.get("legitimacy_bundle") or {}) if isinstance(ctx, dict) else {}
    rein = (ctx.get("reinforcement") or {}) if isinstance(ctx, dict) else {}
    l1 = (ctx.get("layer1_ml") or {}) if isinstance(ctx, dict) else {}
    no_free = bool(bundle.get("no_free_hosting_signal"))
    no_mismatch = bool(bundle.get("no_brand_mismatch"))
    no_susp_form = not bool(bundle.get("suspicious_form_action_cross_origin"))
    cap = rein if isinstance(rein, dict) else {}
    redir = int(cap.get("redirect_count") or 0)
    xredir = int(cap.get("cross_domain_redirect_count") or 0)
    no_susp_redirect = redir <= 1 and xredir <= 0
    raw = l1.get("phish_proba_raw")
    try:
        raw_f = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        raw_f = 0.0
    no_strong_phish = raw_f < 0.90
    return all((no_free, no_mismatch, no_susp_form, no_susp_redirect, no_strong_phish))


def _is_high_legit_content_rescue_candidate(
    *,
    pre_ai_score: float,
    ai_result: Dict[str, Any],
    adjudication_context: Optional[Dict[str, Any]],
) -> bool:
    if pre_ai_score < 0.38:
        return False
    if str(ai_result.get("ai_assessment") or "") != "likely_legitimate":
        return False
    if str(ai_result.get("ai_confidence") or "") not in {"medium", "high"}:
        return False
    ctx = adjudication_context or {}
    hp = (ctx.get("host_path_reasoning") or {}) if isinstance(ctx, dict) else {}
    domblk = (ctx.get("html_dom_anomaly") or {}) if isinstance(ctx, dict) else {}
    dom = (domblk.get("summary") or {}) if isinstance(domblk, dict) else {}
    bundle = (ctx.get("legitimacy_bundle") or {}) if isinstance(ctx, dict) else {}
    if str(hp.get("host_legitimacy_confidence") or "") != "high":
        return False
    if str(hp.get("path_fit_assessment") or "") not in {"plausible", "unusual_but_possible"}:
        return False
    family = str(dom.get("page_family") or "")
    if family not in {"content_feed_forum_aggregator", "article_news", "public_docs_or_reference"}:
        return False
    blocked = any(
        (
            bool(dom.get("trust_action_context")),
            bool(dom.get("suspicious_credential_collection_pattern")),
            int(dom.get("form_action_external_domain_count") or 0) > 0,
            bool(dom.get("login_harvester_pattern")),
            bool(dom.get("wrapper_page_pattern") or dom.get("interstitial_or_preview_pattern")),
            bool(dom.get("trust_surface_brand_domain_mismatch")),
            int(dom.get("anchor_strong_mismatch_count") or 0) > 0,
            str(hp.get("host_identity_class") or "") == "suspicious_host_pattern",
            str(hp.get("path_fit_assessment") or "") == "suspicious",
            bool(bundle.get("suspicious_form_action_cross_origin")),
            not bool(bundle.get("no_deceptive_token_placement", True)),
            not bool(bundle.get("no_free_hosting_signal", True)),
        )
    )
    return not blocked


def apply_ai_adjustment(
    *,
    pre_ai_score: Optional[float],
    pre_ai_verdict: str,
    ai_result: Dict[str, Any],
    adjudication_context: Optional[Dict[str, Any]] = None,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
    cfg: Optional[AIAdjudicationConfig] = None,
) -> Dict[str, Any]:
    if pre_ai_score is None:
        return {
            "applied": False,
            "pre_ai_score": None,
            "post_ai_score": None,
            "adjustment_applied": 0.0,
            "post_ai_verdict": pre_ai_verdict,
            "reason": "no_pre_ai_score",
        }
    c = cfg or AIAdjudicationConfig()
    vcfg = verdict_cfg or Verdict3WayConfig()
    s0 = float(pre_ai_score)
    direction = str(ai_result.get("adjustment_direction") or "none")
    ai_assessment = str(ai_result.get("ai_assessment") or "uncertain")
    # Keep adjustment direction consistent with adjudicated class.
    if ai_assessment == "likely_legitimate" and direction == "up":
        direction = "down"
    elif ai_assessment == "likely_phishing" and direction == "down":
        direction = "up"
    elif ai_assessment == "uncertain" and direction not in {"none"}:
        direction = "none"
    mag_req = float(ai_result.get("adjustment_magnitude") or 0.0)
    mag = _bounded_magnitude(s0, mag_req, c)

    narrow_boost_applied = False
    content_rescue_boost_applied = False
    if direction == "down" and _is_borderline_legit_candidate(
        pre_ai_score=s0,
        pre_ai_verdict=pre_ai_verdict,
        ai_result=ai_result,
        adjudication_context=adjudication_context,
    ):
        boosted_req = min(c.narrow_legit_down_max_adjustment, mag_req * 1.35)
        boosted_center_weight = max(0.85, 1.0 - min(1.0, abs(s0 - 0.5) / 0.35))
        mag = min(c.narrow_legit_down_max_adjustment, boosted_req * boosted_center_weight)
        narrow_boost_applied = True
    elif direction == "down" and _is_high_legit_content_rescue_candidate(
        pre_ai_score=s0,
        ai_result=ai_result,
        adjudication_context=adjudication_context,
    ):
        boosted_req = min(c.high_legit_content_down_max_adjustment, max(mag_req, 0.10) * 1.25)
        boosted_center_weight = max(0.80, 1.0 - min(1.0, abs(s0 - 0.5) / 0.45))
        mag = min(c.high_legit_content_down_max_adjustment, boosted_req * boosted_center_weight)
        content_rescue_boost_applied = True

    review = (
        (adjudication_context or {}).get("ml_phishing_capture_miss_review")
        if isinstance(adjudication_context, dict)
        else None
    )
    block_legit = bool((review or {}).get("block_ai_legitimizing_adjustment"))

    delta = 0.0
    if direction == "down":
        delta = -mag
    elif direction == "up":
        delta = mag
    if block_legit and delta < 0:
        delta = 0.0

    s1 = min(1.0, max(0.0, s0 + delta))

    # Guardrails: do not let AI flip obviously strong zones by itself.
    if s0 >= c.strong_high_guard and s1 < vcfg.combined_high:
        if content_rescue_boost_applied:
            # For high-legitimacy content rescue, still bounded but allow reaching uncertain.
            s1 = max(vcfg.combined_low, s0 - c.high_legit_content_down_max_adjustment)
        else:
            s1 = max(vcfg.combined_high, s0 - c.max_adjustment * 0.35)
    if s0 <= c.strong_low_guard and s1 > vcfg.combined_low:
        s1 = min(vcfg.combined_low, s0 + c.max_adjustment * 0.35)

    if block_legit:
        post_chk, _ = verdict_3way(s1, vcfg)
        if post_chk == "likely_legitimate":
            lo, hi = float(vcfg.combined_low), float(vcfg.combined_high)
            s1 = min(hi - 1e-3, max(lo + 1e-3, (lo + hi) / 2.0))

    post_v, post_why = verdict_3way(s1, vcfg)
    return {
        "applied": abs(s1 - s0) > 1e-8,
        "pre_ai_score": s0,
        "post_ai_score": float(s1),
        "adjustment_applied": float(s1 - s0),
        "narrow_legit_down_boost_applied": narrow_boost_applied,
        "high_legit_content_down_boost_applied": content_rescue_boost_applied,
        "post_ai_verdict": post_v,
        "post_ai_rule": post_why,
        "pre_ai_verdict": pre_ai_verdict,
    }


def run_ai_adjudication(
    *,
    input_url: str,
    layer1_ml: Dict[str, Any],
    reinforcement: Optional[Dict[str, Any]],
    verdict_pre_ai: Dict[str, Any],
    legitimacy_bundle: Dict[str, Any],
    html_structure_summary: Optional[Dict[str, Any]],
    html_structure_risk_score: Optional[float],
    html_structure_reasons: Optional[List[str]],
    html_dom_anomaly_summary: Optional[Dict[str, Any]] = None,
    html_dom_anomaly_risk_score: Optional[float] = None,
    html_dom_anomaly_reasons: Optional[List[str]] = None,
    html_dom_visual_assessment: Optional[str] = None,
    host_path_reasoning: Optional[Dict[str, Any]] = None,
    evidence_gaps: List[str],
    capture_json: Optional[Dict[str, Any]],
    enabled: bool,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
    html_capture_missing_reason: Optional[str] = None,
    html_structure_error: Optional[str] = None,
    html_dom_anomaly_error: Optional[str] = None,
    pre_verdict_before_ml_capture_miss_safety: Optional[str] = None,
) -> Dict[str, Any]:
    base_score = verdict_pre_ai.get("combined_score")
    pre_label = str(verdict_pre_ai.get("verdict_3way") or "uncertain")
    if not enabled:
        return {"enabled": False, "ran": False, "skip_reason": "disabled"}
    cap = (reinforcement or {}).get("capture") or {}
    legit_snapshot = (
        str(pre_verdict_before_ml_capture_miss_safety)
        if pre_verdict_before_ml_capture_miss_safety is not None
        else pre_label
    )
    force_ml_capture_miss_review = (
        bool(layer1_ml.get("predicted_phishing"))
        and legit_snapshot == "likely_legitimate"
        and bool(cap.get("capture_failed"))
        and _html_missing_for_ml_capture_miss_review(
            html_capture_missing_reason=html_capture_missing_reason,
            html_structure_error=html_structure_error,
            html_dom_anomaly_error=html_dom_anomaly_error,
        )
        and not _trust_anchor_present(legitimacy_bundle)
    )
    should, reasons = should_run_ai_adjudication(
        pre_ai_combined=float(base_score) if base_score is not None else None,
        ml_effective_score=verdict_pre_ai.get("effective_ml_score"),
        org_risk_adjusted=float(verdict_pre_ai.get("org_risk_adjusted") or 0.0),
        bundle=legitimacy_bundle,
        pre_verdict=pre_label,
        input_url=input_url,
        html_dom_anomaly_summary=html_dom_anomaly_summary,
        host_path_reasoning=host_path_reasoning,
        verdict_cfg=verdict_cfg,
        force_ml_phishing_capture_miss_review=force_ml_capture_miss_review,
    )
    if not should:
        return {"enabled": True, "ran": False, "skip_reason": "not_eligible", "eligibility_reasons": reasons}

    miss_review = {
        "predicted_phishing": bool(layer1_ml.get("predicted_phishing")),
        "capture_failed": cap.get("capture_failed"),
        "capture_failure_type": cap.get("capture_failure_type"),
        "html_capture_missing_reason": html_capture_missing_reason,
        "html_structure_error": html_structure_error,
        "html_dom_anomaly_error": html_dom_anomaly_error,
        "official_registrable_anchor": legitimacy_bundle.get("official_registrable_anchor"),
        "official_domain_family": legitimacy_bundle.get("official_domain_family"),
        "strong_trust_anchor": legitimacy_bundle.get("strong_trust_anchor"),
        "block_ai_legitimizing_adjustment": force_ml_capture_miss_review,
    }
    packet = _build_evidence_packet(
        input_url=input_url,
        layer1_ml=layer1_ml,
        reinforcement=reinforcement,
        verdict_pre_ai=verdict_pre_ai,
        legitimacy_bundle=legitimacy_bundle,
        html_structure_summary=html_structure_summary,
        html_structure_risk_score=html_structure_risk_score,
        html_structure_reasons=html_structure_reasons,
        html_dom_anomaly_summary=html_dom_anomaly_summary,
        html_dom_anomaly_risk_score=html_dom_anomaly_risk_score,
        html_dom_anomaly_reasons=html_dom_anomaly_reasons,
        html_dom_visual_assessment=html_dom_visual_assessment,
        host_path_reasoning=host_path_reasoning,
        evidence_gaps=evidence_gaps,
        capture_json=capture_json,
        html_capture_missing_reason=html_capture_missing_reason,
        html_structure_error=html_structure_error,
        html_dom_anomaly_error=html_dom_anomaly_error,
        ml_phishing_capture_miss_review=miss_review,
    )
    ai_result = call_ai_adjudicator(packet)
    if not ai_result.get("ran"):
        return {
            "enabled": True,
            "ran": False,
            "skip_reason": "ai_call_unavailable",
            "eligibility_reasons": reasons,
            "ai_error": ai_result.get("ai_error"),
        }
    applied = apply_ai_adjustment(
        pre_ai_score=float(base_score) if base_score is not None else None,
        pre_ai_verdict=pre_label,
        ai_result=ai_result,
        adjudication_context=packet,
        verdict_cfg=verdict_cfg,
    )
    return {
        "enabled": True,
        "ran": True,
        "eligibility_reasons": reasons,
        "evidence_packet": packet,
        "ai_result": ai_result,
        "adjustment": applied,
    }
