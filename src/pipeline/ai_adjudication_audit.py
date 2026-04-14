"""Compact audit for AI adjudication effects across evaluation suites."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.evaluation_sets import DEFAULT_URL_SUITES_JSON, load_url_suites
from src.pipeline.paths import ensure_layout, reports_dir

logger = logging.getLogger(__name__)

_JSON_NAME = "ai_adjudication_audit.json"
_MD_NAME = "ai_adjudication_audit.md"

_LEGIT_SUITES = {"obvious_legit", "tricky_legit"}
_PHISH_SUITES = {"obvious_phish", "hard_phishing"}


def _transition(pre_v: str, post_v: str) -> str:
    return f"{pre_v}->{post_v}"


def _score_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _extract_pre_post(verdict: Dict[str, Any], ai_block: Dict[str, Any]) -> Dict[str, Any]:
    pre_score = _score_or_none(verdict.get("combined_score_pre_ai"))
    post_score = _score_or_none(verdict.get("combined_score"))
    pre_verdict = str(verdict.get("verdict_3way_pre_ai") or verdict.get("verdict_3way") or "uncertain")
    post_verdict = str(verdict.get("verdict_3way") or pre_verdict)

    adj = ai_block.get("adjustment") or {}
    if pre_score is None:
        pre_score = _score_or_none(adj.get("pre_ai_score")) or post_score
    if post_score is None:
        post_score = _score_or_none(adj.get("post_ai_score")) or pre_score
    if "pre_ai_verdict" in adj and adj.get("pre_ai_verdict"):
        pre_verdict = str(adj.get("pre_ai_verdict"))
    if "post_ai_verdict" in adj and adj.get("post_ai_verdict"):
        post_verdict = str(adj.get("post_ai_verdict"))

    return {
        "pre_score": pre_score,
        "post_score": post_score,
        "pre_verdict": pre_verdict,
        "post_verdict": post_verdict,
    }


def audit_row(
    *,
    suite: str,
    url: str,
    reinforcement: bool,
    layer1_use_dns: bool,
) -> Dict[str, Any]:
    from src.app_v1.analyze_dashboard import build_dashboard_analysis

    payload, _gaps = build_dashboard_analysis(
        url,
        reinforcement=reinforcement,
        layer1_use_dns=layer1_use_dns,
        ai_adjudication=True,
    )
    verdict = payload.get("verdict") or {}
    ai_block = verdict.get("ai_adjudication") or {}
    ai_result = ai_block.get("ai_result") or {}
    adj = ai_block.get("adjustment") or {}
    prepost = _extract_pre_post(verdict, ai_block)

    pre_score = prepost["pre_score"]
    post_score = prepost["post_score"]
    pre_v = prepost["pre_verdict"]
    post_v = prepost["post_verdict"]
    direction = str(ai_result.get("adjustment_direction") or "none")
    magnitude = float(ai_result.get("adjustment_magnitude") or 0.0)
    applied_delta = _score_or_none(adj.get("adjustment_applied"))

    strong_phish_soften_attempt = bool(
        suite in _PHISH_SUITES
        and pre_score is not None
        and pre_score >= 0.85
        and (direction == "down" or (applied_delta is not None and applied_delta < -1e-6))
    )
    strong_legit_raise_attempt = bool(
        suite in _LEGIT_SUITES
        and pre_score is not None
        and pre_score <= 0.15
        and (direction == "up" or (applied_delta is not None and applied_delta > 1e-6))
    )
    ai_ran_should_skip = bool(ai_block.get("ran") and not (ai_block.get("eligibility_reasons") or []))

    return {
        "suite": suite,
        "url": url,
        "pre_ai_combined_score": pre_score,
        "post_ai_combined_score": post_score,
        "pre_ai_verdict": pre_v,
        "post_ai_verdict": post_v,
        "transition": _transition(pre_v, post_v),
        "ai_ran": bool(ai_block.get("ran")),
        "ai_enabled": bool(ai_block.get("enabled", True)),
        "ai_skip_reason": ai_block.get("skip_reason"),
        "eligibility_reasons": ai_block.get("eligibility_reasons") or [],
        "adjustment_direction": direction,
        "adjustment_magnitude_requested": magnitude,
        "adjustment_applied_delta": applied_delta if applied_delta is not None else 0.0,
        "ai_assessment": ai_result.get("ai_assessment"),
        "ai_confidence": ai_result.get("ai_confidence"),
        "ai_summary": ai_result.get("summary"),
        "strong_phish_soften_attempt": strong_phish_soften_attempt,
        "strong_legit_raise_attempt": strong_legit_raise_attempt,
        "ai_ran_when_should_skip_flag": ai_ran_should_skip,
    }


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def cnt(a: str, b: str) -> int:
        return sum(1 for r in rows if r.get("pre_ai_verdict") == a and r.get("post_ai_verdict") == b)

    return {
        "n": len(rows),
        "ai_ran_count": sum(1 for r in rows if r.get("ai_ran")),
        "uncertain_to_likely_legitimate": cnt("uncertain", "likely_legitimate"),
        "uncertain_to_likely_phishing": cnt("uncertain", "likely_phishing"),
        "likely_legitimate_to_uncertain": cnt("likely_legitimate", "uncertain"),
        "likely_phishing_to_uncertain": cnt("likely_phishing", "uncertain"),
        "strong_phish_soften_attempt_count": sum(1 for r in rows if r.get("strong_phish_soften_attempt")),
        "strong_legit_raise_attempt_count": sum(1 for r in rows if r.get("strong_legit_raise_attempt")),
        "ai_ran_when_should_skip_count": sum(1 for r in rows if r.get("ai_ran_when_should_skip_flag")),
    }


def run_ai_adjudication_audit(
    *,
    suites_json: Optional[Path] = None,
    reinforcement: bool = False,
    layer1_use_dns: bool = False,
) -> Dict[str, Any]:
    ensure_layout()
    suites = load_url_suites(suites_json)
    wanted = ("obvious_legit", "tricky_legit", "obvious_phish", "hard_phishing")
    rows: List[Dict[str, Any]] = []
    for s in wanted:
        for url in suites.get(s, []):
            try:
                rows.append(
                    audit_row(
                        suite=s,
                        url=str(url),
                        reinforcement=reinforcement,
                        layer1_use_dns=layer1_use_dns,
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("ai adjudication audit row failed")
                rows.append({"suite": s, "url": str(url), "error": f"{type(e).__name__}: {e}"})

    per_suite = {s: _summary([r for r in rows if r.get("suite") == s]) for s in wanted}
    suspicious_flags = {
        "strong_phish_softened": [
            {"suite": r.get("suite"), "url": r.get("url"), "transition": r.get("transition")}
            for r in rows
            if r.get("strong_phish_soften_attempt")
        ],
        "strong_legit_raised": [
            {"suite": r.get("suite"), "url": r.get("url"), "transition": r.get("transition")}
            for r in rows
            if r.get("strong_legit_raise_attempt")
        ],
        "ai_ran_when_should_skip": [
            {"suite": r.get("suite"), "url": r.get("url"), "eligibility_reasons": r.get("eligibility_reasons")}
            for r in rows
            if r.get("ai_ran_when_should_skip_flag")
        ],
    }
    report = {
        "audit_type": "ai_adjudication_compact",
        "reinforcement_enabled": reinforcement,
        "suites_source": str(suites_json or DEFAULT_URL_SUITES_JSON),
        "summary": _summary(rows),
        "per_suite": per_suite,
        "suspicious_ai_behavior": suspicious_flags,
        "rows": rows,
    }
    out_json = reports_dir() / _JSON_NAME
    out_md = reports_dir() / _MD_NAME
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_to_md(report), encoding="utf-8")
    return report


def _to_md(report: Dict[str, Any]) -> str:
    s = report.get("summary") or {}
    lines = [
        "# AI Adjudication Audit",
        "",
        f"- Reinforcement enabled: **{report.get('reinforcement_enabled')}**",
        f"- Suites source: `{report.get('suites_source')}`",
        "",
        "## Overall",
        "",
        f"- n={s.get('n', 0)}, ai_ran={s.get('ai_ran_count', 0)}",
        f"- uncertain -> likely_legitimate: **{s.get('uncertain_to_likely_legitimate', 0)}**",
        f"- uncertain -> likely_phishing: **{s.get('uncertain_to_likely_phishing', 0)}**",
        f"- likely_legitimate -> uncertain: **{s.get('likely_legitimate_to_uncertain', 0)}**",
        f"- likely_phishing -> uncertain: **{s.get('likely_phishing_to_uncertain', 0)}**",
        f"- strong-phish soften attempts: **{s.get('strong_phish_soften_attempt_count', 0)}**",
        f"- strong-legit raise attempts: **{s.get('strong_legit_raise_attempt_count', 0)}**",
        f"- AI ran when should skip: **{s.get('ai_ran_when_should_skip_count', 0)}**",
        "",
        "## Per suite",
        "",
    ]
    for suite, ss in (report.get("per_suite") or {}).items():
        lines.append(
            f"- `{suite}`: n={ss.get('n', 0)}, ai_ran={ss.get('ai_ran_count', 0)}, "
            f"u->legit={ss.get('uncertain_to_likely_legitimate', 0)}, "
            f"u->phish={ss.get('uncertain_to_likely_phishing', 0)}, "
            f"phish->u={ss.get('likely_phishing_to_uncertain', 0)}"
        )
    lines.extend(["", f"JSON: `outputs/reports/{_JSON_NAME}`", ""])
    return "\n".join(lines)


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    setup_logging(logs_dir() / "ai_adjudication_audit.log")
    ap = argparse.ArgumentParser(description="Audit AI adjudication pre/post effects across evaluation suites.")
    ap.add_argument("--suites-json", type=Path, default=None, help=f"Default: {DEFAULT_URL_SUITES_JSON}")
    ap.add_argument("--reinforcement", action="store_true", help="Enable live reinforcement during audit.")
    ap.add_argument("--layer1-use-dns", action="store_true")
    args = ap.parse_args()
    rep = run_ai_adjudication_audit(
        suites_json=args.suites_json,
        reinforcement=args.reinforcement,
        layer1_use_dns=args.layer1_use_dns,
    )
    print(json.dumps({"summary": rep["summary"], "per_suite": rep["per_suite"]}, indent=2))
    print("Wrote", reports_dir() / _JSON_NAME)
    print("Wrote", reports_dir() / _MD_NAME)


if __name__ == "__main__":
    main()
