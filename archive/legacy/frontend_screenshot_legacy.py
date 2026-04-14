"""DEPRECATED: screenshot-first Streamlit UI (legacy reference only).

This file is preserved for historical context and old experiments.
Primary maintained UI is: ``src/app_v1/frontend.py``.

Run legacy UI from project root::

    streamlit run archive/legacy/frontend_screenshot_legacy.py

Requires: pip install streamlit
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

# Resolve imports when launched as ``streamlit run src/app_v1/frontend.py`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

logger = logging.getLogger(__name__)


def _resolve_artifact_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    # Normalize Windows/Unix separators and resolve relative paths from repo root.
    normalized = str(path).strip().replace("\\", "/")
    p = Path(normalized)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def _safe_image(path: Optional[str], caption: str) -> None:
    if not path:
        st.caption(f"No path for {caption}.")
        return
    p = _resolve_artifact_path(path)
    if p is None:
        st.caption(f"No path for {caption}.")
        return
    if not p.is_file():
        st.warning(f"{caption}: file not found ({p})")
        return
    try:
        st.image(str(p), caption=caption, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"{caption}: could not load image ({exc})")


def _fallback_row(url: str, error_message: str) -> Dict[str, Any]:
    return {
        "input_url": url,
        "capture": {},
        "ai_brand_task": {},
        "legit_lookup": {},
        "legit_reference_capture": None,
        "features": {},
        "comparison": {
            "reasons": [
                "Visual similarity was skipped for this URL; deterministic comparison signals were still used."
            ]
        },
        "verdict": {
            "verdict": "inconclusive",
            "confidence": "low",
            "reasons": [
                "Analysis did not complete cleanly. Partial diagnostics are shown below."
            ],
            "error": error_message,
        },
        "error": error_message,
    }


def _capture_provenance_explanation(strategy: Optional[str]) -> str:
    """Short plain-English line for how the suspicious URL was captured."""
    s = (strategy or "").strip().lower()
    if s == "failed":
        return "Capture failed entirely; verdict used non-visual signals"
    if s == "http_fallback":
        return "Browser capture was blocked; fell back to raw HTML only"
    if s == "playwright_stealth":
        return "Initial browser capture was blocked; stealth retry succeeded"
    if s == "playwright_headless":
        return "Captured normally in headless browser"
    return "Capture provenance unavailable (no capture metadata for this run)."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _confidence_band(confidence: Optional[str]) -> tuple[int, str]:
    c = (confidence or "").strip().lower()
    if c == "high":
        return 90, "High confidence"
    if c == "medium":
        return 60, "Medium confidence"
    if c == "low":
        return 30, "Low confidence"
    return 10, "Unknown confidence"


def _risk_spectrum_position(verdict: Optional[str]) -> tuple[int, str]:
    v = (verdict or "").strip().lower()
    if v == "likely_legit":
        return 15, "Likely legit"
    if v == "suspicious":
        return 55, "Suspicious"
    if v == "likely_phishing":
        return 90, "Likely phishing"
    return 50, "Inconclusive"


def _stage_errors(row: Dict[str, Any]) -> Dict[str, str]:
    stage_map: Dict[str, str] = {}
    for stage in [
        "capture",
        "ai_brand_task",
        "legit_lookup",
        "features",
        "comparison",
        "verdict",
    ]:
        payload = row.get(stage)
        if isinstance(payload, dict):
            err = payload.get("error")
            if err:
                stage_map[stage] = str(err)
    if row.get("error"):
        stage_map["pipeline"] = str(row["error"])
    return stage_map


def _extract_json_from_stdout(stdout: str) -> Dict[str, Any]:
    """Parse orchestrator CLI JSON output and return row payload."""
    raw = (stdout or "").strip()
    if not raw:
        raise ValueError("Pipeline subprocess produced empty stdout.")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Be defensive if extra logs are emitted before/after JSON.
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Could not find JSON object in subprocess stdout.")
        obj = json.loads(raw[start : end + 1])
    if isinstance(obj, dict) and isinstance(obj.get("row"), dict):
        return obj["row"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Subprocess JSON output was not an object.")


def _run_pipeline_subprocess(url: str) -> Dict[str, Any]:
    """Run pipeline in separate Python process (avoids Streamlit/Playwright sync conflicts on Windows)."""
    cmd = [sys.executable, "-m", "app_v1.orchestrator", "--url", url]
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT / "src"),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or "No error output from subprocess."
        raise RuntimeError(
            f"Pipeline subprocess failed with exit code {proc.returncode}: {detail}"
        )
    return _extract_json_from_stdout(proc.stdout)


def _show_results(row: Dict[str, Any]) -> None:
    verdict = row.get("verdict") or {}
    ai = row.get("ai_brand_task") or {}
    url_intel = row.get("url_intel") or {}
    comp = row.get("comparison") or {}
    legit_lookup = row.get("legit_lookup") or {}
    cap = row.get("capture") or {}
    feat = row.get("features") or {}
    legit = row.get("legit_reference_capture")

    st.subheader("Verdict")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Verdict", verdict.get("verdict") or "—")
    with c2:
        st.metric("Confidence", verdict.get("confidence") or "—")

    reasons = verdict.get("reasons") or []
    if reasons:
        st.markdown("**Reasons**")
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.caption("No reasons listed.")
    if cap.get("intent_summary"):
        st.info(f"Intent summary: {cap.get('intent_summary')}")
    st.subheader("URL Intelligence")
    st.write(
        {
            "url_brand_hint": url_intel.get("brand_hint"),
            "url_product_hint": url_intel.get("product_hint"),
            "url_action_hint": url_intel.get("action_hint"),
            "url_language_hint": url_intel.get("language_hint"),
            "url_locale_hint": url_intel.get("locale_hint"),
            "url_first_party_plausibility": url_intel.get("first_party_url_plausibility"),
        }
    )
    reasons = url_intel.get("url_shape_reasons") or []
    if reasons:
        st.caption("URL shape reasons: " + "; ".join(str(x) for x in reasons[:4]))

    st.subheader("Classification summary")
    st.write(
        {
            "brand_guess": ai.get("brand_guess"),
            "task_guess": ai.get("task_guess"),
            "trusted_reference_found": comp.get("trusted_reference_found"),
            "final_settled_url": cap.get("final_url"),
            "redirect_count": cap.get("redirect_count"),
            "cross_domain_redirect_count": cap.get("cross_domain_redirect_count"),
            "settled_successfully": cap.get("settled_successfully"),
            "settle_time_ms": cap.get("settle_time_ms"),
            "suspicious_language": cap.get("detected_language"),
            "legit_language": (legit.get("detected_language") if isinstance(legit, dict) else None),
            "language_match": comp.get("language_match"),
            "legit_reference_matches_intended_task": comp.get("legit_reference_matches_intended_task"),
            "legit_reference_quality": comp.get("legit_reference_quality"),
            "legit_reference_match_tier": (
                comp.get("legit_reference_match_tier")
                or legit_lookup.get("legit_reference_match_tier")
            ),
            "url_product_action_aligned": comp.get("url_product_action_aligned"),
        }
    )
    if cap.get("settled_successfully"):
        st.caption("Screenshot captured after reaching settled network state.")
    else:
        st.caption("Screenshot captured in best-effort rendered state (network did not fully settle).")

    st.subheader("Visual Evidence Summary")
    action_match = max(0.0, min(1.0, _safe_float(comp.get("action_match_score"))))
    title_similarity = max(0.0, min(1.0, _safe_float(comp.get("title_similarity"))))
    visible_text_similarity = max(
        0.0, min(1.0, _safe_float(comp.get("visible_text_similarity")))
    )
    dom_similarity = max(0.0, min(1.0, _safe_float(comp.get("dom_similarity_score"))))
    behavior_gap = max(0.0, min(1.0, _safe_float(comp.get("behavior_gap_score"))))
    confidence_pct, confidence_label = _confidence_band(verdict.get("confidence"))

    metrics = [
        (
            "action_match_score",
            action_match,
            "How closely the URL/domain/action aligns with the trusted reference.",
        ),
        (
            "title_similarity",
            title_similarity,
            "How similar the page title is to the trusted reference.",
        ),
        (
            "visible_text_similarity",
            visible_text_similarity,
            "How similar the extracted page text is to the trusted reference.",
        ),
        (
            "dom_similarity_score",
            dom_similarity,
            "How similar the HTML structure is to the trusted reference.",
        ),
        (
            "behavior_gap_score",
            behavior_gap,
            "How much the suspicious page differs from the trusted reference overall.",
        ),
    ]
    for metric_name, metric_val, metric_help in metrics:
        st.markdown(f"**{metric_name}: {metric_val:.2f}**")
        st.progress(int(round(metric_val * 100)))
        st.caption(metric_help)

    st.markdown(f"**overall_verdict_confidence_band: {confidence_label} ({confidence_pct}%)**")
    st.progress(confidence_pct)
    st.caption("How strongly the current verdict is supported by available evidence.")

    st.subheader("Risk spectrum")
    spectrum_pos, spectrum_label = _risk_spectrum_position(verdict.get("verdict"))
    st.markdown(
        f"**Current position: {spectrum_label} ({spectrum_pos} / 100)**  \n"
        "Scale: likely legit -> suspicious -> likely phishing"
    )
    st.progress(spectrum_pos)
    if spectrum_pos >= 80:
        st.error("Risk is currently in the likely phishing range.")
    elif spectrum_pos >= 40:
        st.warning("Risk is currently in the suspicious range.")
    else:
        st.success("Risk is currently in the likely legit range.")

    st.subheader("Capture provenance")
    if cap.get("capture_blocked") or feat.get("capture_blocked"):
        st.warning("⚠️ This page may be actively blocking automated inspection.")

    cap_block_reason = cap.get("capture_block_reason")
    cap_block_evidence = cap.get("capture_block_evidence")
    strategy = cap.get("capture_strategy")
    blocked_raw = cap.get("capture_blocked")
    if blocked_raw is None and isinstance(feat, dict):
        blocked_raw = feat.get("capture_blocked")
    blocked_bool = bool(blocked_raw) if blocked_raw is not None else False
    screenshot_ok = bool(str(cap.get("screenshot_path") or "").strip())
    html_ok = bool(str(cap.get("html_path") or "").strip())
    text_ok = bool(str(cap.get("visible_text") or "").strip())
    st.write(
        {
            "capture_blocked": blocked_bool,
            "capture_strategy": strategy if strategy else "—",
            "capture_block_reason": cap_block_reason if cap_block_reason else "—",
            "capture_block_evidence": cap_block_evidence if cap_block_evidence else "—",
            "Screenshot (viewport) captured": "Yes" if screenshot_ok else "No",
            "HTML captured": "Yes" if html_ok else "No",
            "Visible text extracted": "Yes" if text_ok else "No",
        }
    )
    if cap.get("capture_blocked"):
        st.info(
            "Missing screenshots/HTML in this run can be caused by target resistance/evasion, not only by app/runtime issues."
        )
    st.markdown(
        f"**How captured:** {_capture_provenance_explanation(str(strategy) if strategy else None)}"
    )

    st.subheader("Comparison metrics")
    st.write(
        {
            "action_match_score": comp.get("action_match_score"),
            "title_similarity": comp.get("title_similarity"),
            "visible_text_similarity": comp.get("visible_text_similarity"),
            "dom_similarity_score": comp.get("dom_similarity_score"),
            "behavior_gap_score": comp.get("behavior_gap_score"),
            "post_submit_left_trusted_domain": comp.get("post_submit_left_trusted_domain"),
            "trusted_oauth_redirect": comp.get("trusted_oauth_redirect"),
            "language_match": comp.get("language_match"),
            "redirect_count": cap.get("redirect_count"),
            "cross_domain_redirect_count": cap.get("cross_domain_redirect_count"),
            "legit_reference_quality": comp.get("legit_reference_quality"),
            "legit_reference_match_tier": (
                comp.get("legit_reference_match_tier")
                or legit_lookup.get("legit_reference_match_tier")
            ),
        }
    )

    err = row.get("error")
    if err:
        st.error(f"Pipeline reported an error: {err}")
    stage_errors = _stage_errors(row)
    if stage_errors:
        st.subheader("Stage diagnostics")
        for stage, message in stage_errors.items():
            st.warning(f"{stage}: {message}")
    if (comp.get("visual_similarity_score") in (None, 0, 0.0)) and not comp.get("error"):
        st.info(
            "Visual similarity was skipped for this URL; deterministic comparison signals were still used."
        )

    st.subheader("Captured page comparison")
    tier = (
        comp.get("legit_reference_match_tier")
        or legit_lookup.get("legit_reference_match_tier")
        or "unknown"
    )
    if tier == "exact_surface":
        st.success("Reference note: exact surface match.")
    elif tier == "same_product":
        st.info("Reference note: same product family, close surface match.")
    elif tier == "same_brand_fallback":
        st.warning("Reference note: broader same-brand fallback (not exact product surface).")
    elif tier == "weak_fallback":
        st.warning("Reference note: weak fallback reference; comparison confidence is reduced.")
    else:
        st.caption("Reference note: match tier unknown.")
    quality = comp.get("legit_reference_quality") or "unknown"
    if quality in {"error_page", "interstitial", "partial"}:
        st.warning(
            f"Reference quality note: legit reference appears degraded ({quality}); interpretation should be cautious."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Suspicious page (captured)**")
        suspicious_img = cap.get("screenshot_path")
        if suspicious_img:
            _safe_image(suspicious_img, "Suspicious page (captured)")
        else:
            if cap.get("capture_blocked"):
                st.info("Suspicious screenshot unavailable because the target resisted automated capture.")
            else:
                st.info("Suspicious screenshot unavailable for this run.")
    with col_b:
        st.markdown("**Legit reference page (captured)**")
        legit_cap = legit if isinstance(legit, dict) else {}
        legit_img = legit_cap.get("screenshot_path")
        if legit_img:
            _safe_image(legit_img, "Legit reference page (captured)")
        else:
            st.info("No trusted reference screenshot available.")

    with st.expander("Raw JSON (full pipeline row)"):
        st.code(json.dumps(row, ensure_ascii=False, indent=2), language="json")


def main() -> None:
    st.set_page_config(page_title="Phishing Triage", layout="wide")
    st.title("Phishing Triage")
    st.caption("Runs the full `app_v1` pipeline (capture → AI → compare → verdict).")
    if os.getenv("OPENAI_API_KEY"):
        st.success("AI-assisted analysis enabled")
    else:
        st.warning(
            "AI analysis is disabled (missing API key). Results are based on deterministic signals only."
        )

    url = st.text_input("URL to analyze", placeholder="https://example.com", key="url_input")

    if st.button("Analyze", type="primary"):
        url = (url or "").strip()
        if not url:
            st.warning("Please enter a URL.")
            return
        row: Optional[Dict[str, Any]] = None
        with st.spinner("Running pipeline (may take a minute)…"):
            try:
                row = _run_pipeline_subprocess(url)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed: {str(exc)}")
                logger.error(
                    "frontend analyze failed for url=%s with exception=%s",
                    url,
                    exc,
                )
                logger.error("Traceback:\n%s", traceback.format_exc())
                row = _fallback_row(url, f"frontend_exception: {type(exc).__name__}: {exc}")
        if row is not None:
            cache = st.session_state.setdefault("pipeline_results_by_url", {})
            cache[url] = row
            _show_results(row)
        else:
            cached = st.session_state.get("pipeline_results_by_url", {}).get(url)
            if cached is not None:
                st.warning(
                    "This run failed with an exception. Showing the last successful result for this URL "
                    "from the current session (partial / stale view)."
                )
                _show_results(cached)


if __name__ == "__main__":
    main()
