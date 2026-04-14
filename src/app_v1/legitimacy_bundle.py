"""Structured legitimacy signals for reinforcement and verdict policy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.pipeline.features.brand_signals import host_on_official_brand_apex
from src.pipeline.safe_url import safe_hostname


def build_legitimacy_bundle(
    ml_brand_snapshot: Dict[str, Any],
    final_url: str = "",
    input_url: str = "",
    capture_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive a structured bundle from Layer-1 brand-structure features and URLs."""
    snap = ml_brand_snapshot or {}
    official_anchor = bool(snap.get("official_registrable_anchor"))
    official_family = bool(snap.get("official_domain_family"))
    path_brand_miss = bool(snap.get("path_brand_without_official_host"))
    free_host = bool(snap.get("brand_on_free_hosting"))
    deceptive = bool(
        int(snap.get("brand_hyphenated_deception_label", 0) or 0)
        or int(snap.get("brand_typosquat_embedded_in_label", 0) or 0)
    )

    fu = (final_url or input_url or "").strip()
    host = ""
    try:
        host = (urlparse(fu).netloc or "").lower()
    except Exception:
        host = ""

    h_apex, _ = safe_hostname(fu)
    apex_url = bool(h_apex and host_on_official_brand_apex(h_apex))

    suspicious_form_action = suspicious_form_action_hint(capture_json)

    no_brand_mismatch = not path_brand_miss and not free_host
    no_free_hosting = not free_host
    no_deceptive_tokens = not deceptive

    strong_trust = bool(
        (official_anchor or official_family)
        and no_brand_mismatch
        and no_free_hosting
        and no_deceptive_tokens
        and apex_url
        and not suspicious_form_action
    )

    return {
        "official_registrable_anchor": official_anchor,
        "official_domain_family": official_family,
        "host_on_official_brand_apex": apex_url,
        "no_brand_mismatch": no_brand_mismatch,
        "no_free_hosting_signal": no_free_hosting,
        "no_deceptive_token_placement": no_deceptive_tokens,
        "suspicious_form_action_cross_origin": suspicious_form_action,
        "strong_trust_anchor": strong_trust,
        "final_host": host,
    }


def adjust_org_risk_for_legitimacy(
    org_risk: float,
    bundle: Dict[str, Any],
    org_reasons: List[str],
) -> Tuple[float, List[str]]:
    """Lower org-style risk when strong legitimacy anchors hold."""
    r = float(org_risk)
    extra: List[str] = []
    if bundle.get("strong_trust_anchor") and r > 0.0:
        delta = min(0.22, r * 0.55)
        r = max(0.0, r - delta)
        extra.append(
            f"Legitimacy anchor discount applied (-{delta:.2f} org risk): official registrable host, no brand mismatch, no free-hosting/deceptive-token flags."
        )
    return r, list(org_reasons) + extra


def blend_ml_phish_for_legitimacy(
    phish_proba: float,
    org_risk: float,
    bundle: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Optionally reduce effective ML phishing probability when trust bundle is strong."""
    p = float(phish_proba)
    meta: Dict[str, Any] = {"ml_discount_applied": False, "ml_discount": 0.0}
    if not bundle.get("strong_trust_anchor"):
        return p, meta
    if org_risk > 0.35:
        return p, meta
    # Mild pull-down: only when org reinforcement is not already screaming risk.
    discount = 0.18 if org_risk < 0.12 else 0.10
    p2 = max(0.0, p * (1.0 - discount))
    meta["ml_discount_applied"] = True
    meta["ml_discount"] = discount
    return p2, meta


def suspicious_form_action_hint(capture_json: Optional[Dict[str, Any]]) -> bool:
    """Heuristic: if capture exposes HTML path, flag cross-origin form actions (best-effort)."""
    if not capture_json:
        return False
    html_path = capture_json.get("html_path")
    if not html_path:
        return False
    try:
        from pathlib import Path

        p = Path(str(html_path))
        if not p.is_file():
            return False
        text = p.read_text(encoding="utf-8", errors="ignore")[:800_000]
    except OSError:
        return False
    low = text.lower()
    if "<form" not in low:
        return False
    # Very coarse: external action attribute to non-data URL
    import re

    for m in re.finditer(r'<form[^>]+action\s*=\s*["\']([^"\']+)["\']', low):
        act = (m.group(1) or "").strip()
        if not act or act.startswith("#") or act.lower().startswith("javascript:"):
            continue
        if act.startswith("http") and "://" in act:
            return True
    return False
