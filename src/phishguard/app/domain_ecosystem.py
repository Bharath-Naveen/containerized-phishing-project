"""Generalized host/domain relation helpers for ecosystem-aware link trust."""

from __future__ import annotations

import re
from typing import Dict, List

import tldextract


def _registrable(host: str) -> str:
    ext = tldextract.extract((host or "").lower())
    return ".".join(p for p in (ext.domain, ext.suffix) if p).lower()


def _domain_label(registrable: str) -> str:
    ext = tldextract.extract((registrable or "").lower())
    return (ext.domain or "").lower()


def _tokenize(label: str) -> List[str]:
    toks = [t for t in re.split(r"[^a-z0-9]+", (label or "").lower()) if t]
    return [t for t in toks if len(t) >= 4]


def _longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _suspicious_host_shape(host: str) -> bool:
    h = (host or "").lower()
    if not h:
        return True
    if "xn--" in h or "@" in h or h.count("-") >= 4:
        return True
    return False


def domain_relation(source_host: str, target_host: str) -> Dict[str, bool]:
    """Classify source→target relation for trust grouping."""
    sh = (source_host or "").lower().strip(".")
    th = (target_host or "").lower().strip(".")
    if not sh or not th:
        return {
            "same_host": False,
            "same_registrable_domain": False,
            "same_trusted_ecosystem": False,
            "truly_external": True,
        }

    same_host = sh == th
    sreg = _registrable(sh)
    treg = _registrable(th)
    same_reg = bool(sreg and treg and sreg == treg)

    same_ecosystem = same_reg
    if not same_ecosystem and sreg and treg and not (_suspicious_host_shape(sh) or _suspicious_host_shape(th)):
        sdom = _domain_label(sreg)
        tdom = _domain_label(treg)
        if sdom and tdom:
            lcp = _longest_common_prefix(sdom, tdom)
            overlap = set(_tokenize(sdom)) & set(_tokenize(tdom))
            # Conservative cross-registrable ecosystem heuristic for public sibling properties.
            if lcp >= 4 and len(sdom) >= 7 and len(tdom) >= 7:
                same_ecosystem = True
            elif any(len(tok) >= 5 for tok in overlap):
                same_ecosystem = True

    truly_external = not (same_host or same_reg or same_ecosystem)
    return {
        "same_host": same_host,
        "same_registrable_domain": same_reg,
        "same_trusted_ecosystem": same_ecosystem,
        "truly_external": truly_external,
    }
