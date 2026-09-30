"""Compact audit for deterministic evidence adjudication across evaluation suites."""

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

_JSON_NAME = "evidence_adjudication_audit.json"
_MD_NAME = "evidence_adjudication_audit.md"

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


def _extract_pre_post(verdict: Dict[str, Any]) -> Dict[str, Any]:
    score = _score_or_none(verdict.get("combined_score"))
    v = str(verdict.get("verdict_3way") or "uncertain")
    return {"pre_score": score, "post_score": score, "pre_verdict": v, "post_verdict": v}


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
    )
    verdict = payload.get("verdict") or {}
    prepost = _extract_pre_post(verdict)

    pre_score = prepost["pre_score"]
    post_score = prepost["post_score"]
    pre_v = prepost["pre_verdict"]
    post_v = prepost["post_verdict"]
    _ = pre_score

    return {
        "suite": suite,
        "url": url,
        "combined_score": post_score,
        "verdict_3way": post_v,
        "transition": _transition(pre_v, post_v),
        "evidence_phishing_score": verdict.get("evidence_phishing_score"),
        "evidence_legitimacy_score": verdict.get("evidence_legitimacy_score"),
        "evidence_hard_blockers": verdict.get("evidence_hard_blockers") or [],
        "evidence_adjudication_reasons": verdict.get("evidence_adjudication_reasons") or [],
    }


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def cnt(a: str, b: str) -> int:
        return sum(1 for r in rows if r.get("pre_ai_verdict") == a and r.get("post_ai_verdict") == b)

    return {
        "n": len(rows),
        "uncertain_to_likely_legitimate": cnt("uncertain", "likely_legitimate"),
        "uncertain_to_likely_phishing": cnt("uncertain", "likely_phishing"),
        "likely_legitimate_to_uncertain": cnt("likely_legitimate", "uncertain"),
        "likely_phishing_to_uncertain": cnt("likely_phishing", "uncertain"),
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
                logger.exception("evidence adjudication audit row failed")
                rows.append({"suite": s, "url": str(url), "error": f"{type(e).__name__}: {e}"})

    per_suite = {s: _summary([r for r in rows if r.get("suite") == s]) for s in wanted}
    report = {
        "audit_type": "evidence_adjudication_compact",
        "reinforcement_enabled": reinforcement,
        "suites_source": str(suites_json or DEFAULT_URL_SUITES_JSON),
        "summary": _summary(rows),
        "per_suite": per_suite,
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
        "# Evidence Adjudication Audit",
        "",
        f"- Reinforcement enabled: **{report.get('reinforcement_enabled')}**",
        f"- Suites source: `{report.get('suites_source')}`",
        "",
        "## Overall",
        "",
        f"- n={s.get('n', 0)}",
        f"- uncertain -> likely_legitimate: **{s.get('uncertain_to_likely_legitimate', 0)}**",
        f"- uncertain -> likely_phishing: **{s.get('uncertain_to_likely_phishing', 0)}**",
        f"- likely_legitimate -> uncertain: **{s.get('likely_legitimate_to_uncertain', 0)}**",
        f"- likely_phishing -> uncertain: **{s.get('likely_phishing_to_uncertain', 0)}**",
        "",
        "## Per suite",
        "",
    ]
    for suite, ss in (report.get("per_suite") or {}).items():
        lines.append(
            f"- `{suite}`: n={ss.get('n', 0)}, "
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
    setup_logging(logs_dir() / "evidence_adjudication_audit.log")
    ap = argparse.ArgumentParser(description="Audit deterministic evidence adjudication across evaluation suites.")
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
