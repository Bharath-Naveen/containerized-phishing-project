"""Full-system phishing audit (Layer-1 + reinforcement + verdict), mirror of hard-legit FP audit."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.app_v1.verdict_policy import Verdict3WayConfig, verdict_3way
from src.pipeline.evaluation_sets import DEFAULT_URL_SUITES_JSON, load_url_suites
from src.pipeline.paths import ensure_layout, reports_dir

logger = logging.getLogger(__name__)

# Layer-1 raw P(phish) threshold for "was strongly detected" regression heuristics.
STRONG_LAYER1_RAW_THRESHOLD = 0.85

OUTPUT_JSON = "phish_audit_full_system.json"
OUTPUT_MD = "phish_audit_full_system.md"


def _direction_from_deltas(combined_delta: float, org_delta: float) -> str:
    if org_delta < -0.02 or combined_delta < -0.02:
        return "lowered"
    if org_delta > 0.02 or combined_delta > 0.02:
        return "raised"
    return "unchanged"


def audit_phishing_url(
    url: str,
    suite: str,
    *,
    reinforcement: bool,
    layer1_use_dns: bool,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    from src.app_v1.analyze_dashboard import build_dashboard_analysis

    url = (url or "").strip()
    cfg = verdict_cfg or Verdict3WayConfig()
    payload, evidence_gaps = build_dashboard_analysis(
        url,
        reinforcement=reinforcement,
        layer1_use_dns=layer1_use_dns,
        verdict_cfg=cfg,
    )
    ml = payload.get("layer1_ml") or {}
    verdict = payload.get("verdict") or {}
    rein = payload.get("reinforcement") or {}

    p_raw = ml.get("phish_proba_model_raw")
    p_cal = ml.get("phish_proba_calibrated", p_raw)
    p_after_cap = ml.get("phish_proba")
    pre_cap = ml.get("phish_proba_pre_apex_cap")

    org_raw = float(verdict.get("org_risk_raw") or 0.0)
    org_adj = float(verdict.get("org_risk_adjusted") or 0.0)
    combined_delta = float(verdict.get("reinforcement_combined_delta") or 0.0)
    org_delta = float(verdict.get("org_adjustment_delta") or 0.0)
    direction = _direction_from_deltas(combined_delta, org_delta)

    combined_no_rein = verdict.get("combined_score_without_reinforcement")
    ml_only_verdict: Optional[str] = None
    if combined_no_rein is not None:
        ml_only_verdict = verdict_3way(float(combined_no_rein), cfg)[0]

    final_v = verdict.get("verdict_3way")

    try:
        raw_f = float(p_raw) if p_raw is not None else None
    except (TypeError, ValueError):
        raw_f = None
    layer1_strong = raw_f is not None and raw_f >= STRONG_LAYER1_RAW_THRESHOLD

    bundle = verdict.get("legitimacy_bundle") or {}
    blend = verdict.get("ml_legitimacy_blend") or {}
    legitimacy_reduced_suspicion = bool(
        bundle.get("strong_trust_anchor")
        or blend.get("ml_discount_applied")
        or org_delta < -0.02
    )

    regression_uncertain = bool(layer1_strong and final_v == "uncertain")
    regression_likely_legit = bool(final_v == "likely_legitimate")

    ml_path_was_phishing = ml_only_verdict == "likely_phishing"
    reinforcement_softened_verdict = bool(
        ml_path_was_phishing and final_v in {"uncertain", "likely_legitimate"}
    )

    pulled_down = bool(
        direction == "lowered"
        and final_v in {"uncertain", "likely_legitimate"}
        and (raw_f or 0) >= 0.5
    )

    reasons_pull_down: List[str] = []
    if pulled_down or reinforcement_softened_verdict:
        if blend.get("ml_discount_applied"):
            reasons_pull_down.append(
                f"ML legitimacy blend applied (discount={blend.get('ml_discount', 0):.3f})."
            )
        if org_delta < -0.02:
            reasons_pull_down.append(f"Org risk adjusted down by {-org_delta:.3f} (legitimacy anchor discount).")
        if bundle.get("strong_trust_anchor"):
            reasons_pull_down.append("strong_trust_anchor true in legitimacy_bundle (unexpected on phishing host).")
        if ml.get("official_brand_apex_cap_applied"):
            reasons_pull_down.append("official_brand_apex_cap applied on ML display score.")
        if rein.get("org_style") and isinstance(rein["org_style"], dict):
            for r in (rein["org_style"].get("reasons") or [])[:5]:
                reasons_pull_down.append(str(r))

    return {
        "url": url,
        "suite": suite,
        "phish_proba_model_raw": p_raw,
        "phish_proba_calibrated": p_cal,
        "phish_proba_after_cap_or_blend_input": p_after_cap,
        "phish_proba_pre_apex_cap": pre_cap,
        "official_brand_apex_cap_applied": bool(ml.get("official_brand_apex_cap_applied")),
        "verdict_3way_final": final_v,
        "combined_score": verdict.get("combined_score"),
        "combined_score_without_reinforcement": combined_no_rein,
        "verdict_3way_ml_only_combined_proxy": ml_only_verdict,
        "top_linear_signals": ml.get("top_linear_signals"),
        "org_style_risk_raw": org_raw,
        "org_style_risk_adjusted": org_adj,
        "org_adjustment_delta": org_delta,
        "reinforcement_combined_delta": combined_delta,
        "reinforcement_suspicion_direction_vs_layer1": direction,
        "legitimacy_bundle": bundle,
        "ml_legitimacy_blend": blend,
        "legitimacy_anchor_reduced_suspicion": legitimacy_reduced_suspicion,
        "layer1_raw_strong_detection": layer1_strong,
        "regression_flag_strong_layer1_now_uncertain": regression_uncertain,
        "regression_flag_final_likely_legitimate": regression_likely_legit,
        "regression_flag_ml_path_likely_phish_but_final_softened": reinforcement_softened_verdict,
        "pulled_down_by_reinforcement_or_legitimacy": pulled_down,
        "pulled_down_reasons": reasons_pull_down,
        "reinforcement_block_error": rein.get("error"),
        "capture_error": (rein.get("capture") or {}).get("error"),
        "org_style_block": rein.get("org_style"),
        "evidence_gaps": evidence_gaps,
        "layer1_error": ml.get("error"),
    }


def _suite_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "likely_phishing_count": sum(1 for r in rows if r.get("verdict_3way_final") == "likely_phishing"),
        "uncertain_count": sum(1 for r in rows if r.get("verdict_3way_final") == "uncertain"),
        "likely_legitimate_count": sum(1 for r in rows if r.get("verdict_3way_final") == "likely_legitimate"),
        "regression_uncertain_strong_layer1_count": sum(
            1 for r in rows if r.get("regression_flag_strong_layer1_now_uncertain")
        ),
        "regression_likely_legitimate_count": sum(
            1 for r in rows if r.get("regression_flag_final_likely_legitimate")
        ),
        "ml_path_phish_but_final_softened_count": sum(
            1 for r in rows if r.get("regression_flag_ml_path_likely_phish_but_final_softened")
        ),
    }


def run_phish_audit(
    *,
    suites_json: Optional[Path] = None,
    reinforcement: bool = True,
    layer1_use_dns: bool = False,
    verdict_cfg: Optional[Verdict3WayConfig] = None,
) -> Dict[str, Any]:
    ensure_layout()
    all_suites = load_url_suites(suites_json)
    obvious = [(u, "obvious_phish") for u in all_suites.get("obvious_phish", [])]
    hard = [(u, "hard_phishing") for u in all_suites.get("hard_phishing", [])]
    work = obvious + hard

    out_rows: List[Dict[str, Any]] = []
    for url, suite in work:
        try:
            out_rows.append(
                audit_phishing_url(
                    url,
                    suite,
                    reinforcement=reinforcement,
                    layer1_use_dns=layer1_use_dns,
                    verdict_cfg=verdict_cfg,
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("phish audit row failed")
            out_rows.append(
                {
                    "url": url,
                    "suite": suite,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    by_suite: Dict[str, List[Dict[str, Any]]] = {"obvious_phish": [], "hard_phishing": []}
    for r in out_rows:
        s = r.get("suite")
        if s in by_suite:
            by_suite[s].append(r)

    summary_all = _suite_summary(out_rows)
    per_suite = {k: _suite_summary(v) for k, v in by_suite.items()}

    regressions = {
        "strong_layer1_now_uncertain": [
            {"url": r["url"], "suite": r.get("suite"), "raw": r.get("phish_proba_model_raw")}
            for r in out_rows
            if r.get("regression_flag_strong_layer1_now_uncertain")
        ],
        "final_likely_legitimate": [
            {
                "url": r["url"],
                "suite": r.get("suite"),
                "raw": r.get("phish_proba_model_raw"),
                "reasons": r.get("pulled_down_reasons"),
            }
            for r in out_rows
            if r.get("regression_flag_final_likely_legitimate")
        ],
        "ml_combined_path_likely_phish_but_final_softened": [
            {
                "url": r["url"],
                "suite": r.get("suite"),
                "final_verdict": r.get("verdict_3way_final"),
                "combined_without_reinforcement": r.get("combined_score_without_reinforcement"),
                "pulled_down_reasons": r.get("pulled_down_reasons"),
            }
            for r in out_rows
            if r.get("regression_flag_ml_path_likely_phish_but_final_softened")
        ],
    }

    report: Dict[str, Any] = {
        "audit_type": "phishing_full_system",
        "parallel_to": "outputs/reports/fp_audit_hard_legit.json",
        "suites_source": str(suites_json or DEFAULT_URL_SUITES_JSON),
        "reinforcement_enabled": reinforcement,
        "strong_layer1_raw_threshold": STRONG_LAYER1_RAW_THRESHOLD,
        "summary": summary_all,
        "per_suite": per_suite,
        "regression_checks": regressions,
        "rows": out_rows,
    }

    json_path = reports_dir() / OUTPUT_JSON
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = reports_dir() / OUTPUT_MD
    md_path.write_text(_markdown_summary(report), encoding="utf-8")
    return report


def _markdown_summary(report: Dict[str, Any]) -> str:
    s = report["summary"]
    ps = report.get("per_suite") or {}
    reg = report.get("regression_checks") or {}
    lines = [
        "# Phishing full-system audit",
        "",
        f"- Reinforcement: **{report.get('reinforcement_enabled')}**",
        f"- Strong Layer-1 raw threshold (regression heuristic): **{report.get('strong_layer1_raw_threshold')}**",
        "",
        "## Overall summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| n | {s.get('n', 0)} |",
        f"| likely_phishing | {s.get('likely_phishing_count', 0)} |",
        f"| uncertain | {s.get('uncertain_count', 0)} |",
        f"| likely_legitimate | {s.get('likely_legitimate_count', 0)} |",
        "",
    ]
    for suite_name in ("obvious_phish", "hard_phishing"):
        ss = ps.get(suite_name) or {}
        lines.extend(
            [
                f"## Suite: `{suite_name}`",
                "",
                f"- n={ss.get('n', 0)}, likely_phishing={ss.get('likely_phishing_count', 0)}, "
                f"uncertain={ss.get('uncertain_count', 0)}, likely_legitimate={ss.get('likely_legitimate_count', 0)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Regression flags",
            "",
            f"- Strong Layer-1 but **uncertain** final: **{len(reg.get('strong_layer1_now_uncertain') or [])}**",
            f"- Final **likely_legitimate**: **{len(reg.get('final_likely_legitimate') or [])}**",
            f"- ML-only combined path **likely_phishing** but final softened: **{len(reg.get('ml_combined_path_likely_phish_but_final_softened') or [])}**",
            "",
        ]
    )
    if reg.get("final_likely_legitimate"):
        lines.append("### URLs → likely_legitimate (critical)")
        lines.append("")
        for item in reg["final_likely_legitimate"]:
            lines.append(f"- `{item.get('url')}` ({item.get('suite')})")
        lines.append("")
    if reg.get("strong_layer1_now_uncertain"):
        lines.append("### Strong Layer-1 → uncertain")
        lines.append("")
        for item in reg["strong_layer1_now_uncertain"]:
            lines.append(f"- `{item.get('url')}` raw≈{item.get('raw')}")
        lines.append("")
    if reg.get("ml_combined_path_likely_phish_but_final_softened"):
        lines.append("### ML-combined path likely_phishing but verdict softened")
        lines.append("")
        for item in reg["ml_combined_path_likely_phish_but_final_softened"]:
            lines.append(
                f"- `{item.get('url')}` → final **{item.get('final_verdict')}** "
                f"(combined_no_rein≈{item.get('combined_without_reinforcement')})"
            )
        lines.append("")
    lines.append(f"JSON: `outputs/reports/{OUTPUT_JSON}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    setup_logging(logs_dir() / "phish_audit.log")
    ap = argparse.ArgumentParser(
        description="Full-system phishing audit (same stack as dashboard / hard-legit FP audit)."
    )
    ap.add_argument("--suites-json", type=Path, default=None, help=f"Default: {DEFAULT_URL_SUITES_JSON}")
    ap.add_argument(
        "--no-reinforcement",
        action="store_true",
        help="Skip Playwright capture (faster; not a full mirror of live reinforcement).",
    )
    ap.add_argument("--layer1-use-dns", action="store_true")
    args = ap.parse_args()
    rep = run_phish_audit(
        suites_json=args.suites_json,
        reinforcement=not args.no_reinforcement,
        layer1_use_dns=args.layer1_use_dns,
    )
    print(json.dumps({"summary": rep["summary"], "per_suite": rep["per_suite"]}, indent=2))
    print(json.dumps(rep["regression_checks"], indent=2))
    print("Wrote", reports_dir() / OUTPUT_JSON)
    print("Wrote", reports_dir() / OUTPUT_MD)


if __name__ == "__main__":
    main()
