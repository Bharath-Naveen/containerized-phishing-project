"""False-positive audit for tricky legitimate URLs (Layer-1 + optional reinforcement).

Mirror (phishing URLs, full stack): :mod:`src.pipeline.phish_audit`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.evaluation_sets import DEFAULT_HARD_LEGIT_JSONL, load_hard_legit_rows
from src.pipeline.paths import ensure_layout, reports_dir

logger = logging.getLogger(__name__)


def audit_hard_legit_row(
    row: Dict[str, Any],
    *,
    reinforcement: bool,
    layer1_use_dns: bool,
) -> Dict[str, Any]:
    from src.app_v1.analyze_dashboard import build_dashboard_analysis

    url = str(row.get("url") or "").strip()
    cat = row.get("category", "")
    payload, _gaps = build_dashboard_analysis(
        url,
        reinforcement=reinforcement,
        layer1_use_dns=layer1_use_dns,
    )
    ml = payload.get("layer1_ml") or {}
    verdict = payload.get("verdict") or {}
    rein = payload.get("reinforcement") or {}
    org_raw = float(verdict.get("org_risk_raw") or 0.0)
    org_adj = float(verdict.get("org_risk_adjusted") or 0.0)
    combined_delta = float(verdict.get("reinforcement_combined_delta") or 0.0)
    org_delta = float(verdict.get("org_adjustment_delta") or 0.0)
    direction = "unchanged"
    if org_delta < -0.02 or combined_delta < -0.02:
        direction = "lowered"
    elif org_delta > 0.02 or combined_delta > 0.02:
        direction = "raised"

    return {
        "url": url,
        "category": cat,
        "phish_proba_model_raw": ml.get("phish_proba_model_raw"),
        "phish_proba_calibrated": ml.get("phish_proba_calibrated"),
        "phish_proba_after_cap": ml.get("phish_proba"),
        "verdict_3way": verdict.get("verdict_3way"),
        "combined_score": verdict.get("combined_score"),
        "top_linear_signals": ml.get("top_linear_signals"),
        "legitimacy_bundle": verdict.get("legitimacy_bundle"),
        "ml_legitimacy_blend": verdict.get("ml_legitimacy_blend"),
        "reinforcement_combined_delta": combined_delta,
        "org_adjustment_delta": org_delta,
        "reinforcement_suspicion_direction_vs_layer1": direction,
        "reinforcement_block_error": rein.get("error"),
        "capture_error": (rein.get("capture") or {}).get("error"),
        "org_style_risk_raw": org_raw,
        "org_style_risk_adjusted": org_adj,
    }


def run_hard_legit_audit(
    *,
    jsonl_path: Optional[Path] = None,
    reinforcement: bool = False,
    layer1_use_dns: bool = False,
) -> Dict[str, Any]:
    ensure_layout()
    rows = load_hard_legit_rows(jsonl_path)
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        try:
            out_rows.append(audit_hard_legit_row(row, reinforcement=reinforcement, layer1_use_dns=layer1_use_dns))
        except Exception as e:  # noqa: BLE001
            logger.exception("audit row failed")
            out_rows.append(
                {
                    "url": row.get("url"),
                    "category": row.get("category"),
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    summary = {
        "n": len(out_rows),
        "likely_phishing_count": sum(1 for r in out_rows if r.get("verdict_3way") == "likely_phishing"),
        "uncertain_count": sum(1 for r in out_rows if r.get("verdict_3way") == "uncertain"),
        "likely_legitimate_count": sum(1 for r in out_rows if r.get("verdict_3way") == "likely_legitimate"),
    }
    report = {"suite": "hard_legit", "summary": summary, "rows": out_rows}
    outp = reports_dir() / "fp_audit_hard_legit.json"
    outp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    setup_logging(logs_dir() / "fp_audit.log")
    ap = argparse.ArgumentParser(description="FP audit on curated hard-legit URLs.")
    ap.add_argument("--jsonl", type=Path, default=None, help=f"Default: {DEFAULT_HARD_LEGIT_JSONL}")
    ap.add_argument("--reinforcement", action="store_true", help="Run Playwright capture (slow).")
    ap.add_argument("--layer1-use-dns", action="store_true")
    args = ap.parse_args()
    rep = run_hard_legit_audit(
        jsonl_path=args.jsonl,
        reinforcement=args.reinforcement,
        layer1_use_dns=args.layer1_use_dns,
    )
    print(json.dumps(rep["summary"], indent=2))
    print("Wrote", reports_dir() / "fp_audit_hard_legit.json")


if __name__ == "__main__":
    main()
