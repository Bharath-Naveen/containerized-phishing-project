"""Streamlit phishing analysis dashboard (Layer-1 ML + optional reinforcement).

Run from repo root::

    streamlit run src/app_v1/frontend.py

Legacy screenshot UI: ``archive/legacy/frontend_screenshot_legacy.py``.
"""

from __future__ import annotations

import json
import logging
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


def _run_dashboard_subprocess(url: str, reinforcement: bool, layer1_dns: bool) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "app_v1.analyze_dashboard",
        "--url",
        url,
    ]
    if not reinforcement:
        cmd.append("--no-reinforcement")
    if layer1_dns:
        cmd.append("--layer1-use-dns")
    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT / "src"),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "analyze_dashboard failed")
    return json.loads(proc.stdout.strip())


def _show_dashboard(row: Dict[str, Any]) -> None:
    ml = row.get("layer1_ml") or {}
    rein = row.get("reinforcement") or {}
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
        st.caption(f"Model: `{ml.get('model_path', '')}`")
        st.caption(f"Canonical URL: `{ml.get('canonical_url', '')}`")
        tops = ml.get("top_linear_signals") or []
        if tops:
            with st.expander("Top linear model signals (logistic regression only)"):
                st.dataframe(tops, use_container_width=True)

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

    if st.button("Analyze", type="primary"):
        url = (url or "").strip()
        if not url:
            st.warning("Enter a URL.")
            return
        with st.spinner("Analyzing…"):
            try:
                row = _run_dashboard_subprocess(url, reinforcement, layer1_dns)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed: {exc}")
                logger.error(traceback.format_exc())
                return
        _show_dashboard(row)


if __name__ == "__main__":
    main()
