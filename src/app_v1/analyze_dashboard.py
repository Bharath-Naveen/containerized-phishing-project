"""Layer-1 ML + optional reinforcement capture → dashboard JSON (no screenshot-vs-reference flow)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.paths import analysis_dir, ensure_layout

from .capture import capture_url
from .config import PipelineConfig
from .ml_layer1 import predict_layer1
from .org_style_signals import org_style_from_capture_blob
from .schemas import utc_now_iso

logger = logging.getLogger(__name__)


def _verdict_from_scores(phish_proba: Optional[float], org_risk: float) -> Dict[str, Any]:
    if phish_proba is None:
        return {
            "label": "uncertain",
            "confidence": "low",
            "reasons": ["ML model unavailable; rely on reinforcement signals only."],
        }
    combined = min(1.0, max(0.0, 0.7 * phish_proba + 0.3 * org_risk))
    reasons: List[str] = [
        f"Layer-1 phishing probability ≈ {phish_proba:.3f}.",
        f"Org-style reinforcement score ≈ {org_risk:.3f}.",
    ]
    if combined >= 0.62:
        return {"label": "likely_phishing", "confidence": "medium", "combined_score": combined, "reasons": reasons}
    if combined <= 0.38:
        return {"label": "likely_legitimate", "confidence": "medium", "combined_score": combined, "reasons": reasons}
    return {"label": "uncertain", "confidence": "low", "combined_score": combined, "reasons": reasons}


def analyze_url_dashboard(
    url: str,
    *,
    reinforcement: bool = True,
    layer1_use_dns: bool = False,
) -> Dict[str, Any]:
    ensure_layout()
    url = (url or "").strip()
    evidence_gaps: List[str] = []
    ml = predict_layer1(url, use_dns=layer1_use_dns)
    if ml.get("error"):
        evidence_gaps.append("Layer-1 ML did not produce a score (missing model or feature error).")

    reinforcement_block: Optional[Dict[str, Any]] = None
    org_risk = 0.0
    if reinforcement:
        cfg = PipelineConfig.from_env()
        try:
            cap = capture_url(url, cfg, namespace="suspicious")
            cj = cap.as_json()
            org = org_style_from_capture_blob(cj, url)
            org_risk = float(org.get("org_style_risk_score") or 0.0)
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

    verdict = _verdict_from_scores(ml.get("phish_proba"), org_risk)
    out: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "input_url": url,
        "layer1_ml": ml,
        "reinforcement": reinforcement_block,
        "verdict": verdict,
        "evidence_gaps": evidence_gaps,
    }
    analysis_dir().mkdir(parents=True, exist_ok=True)
    (analysis_dir() / "last_dashboard_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def main() -> None:
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
