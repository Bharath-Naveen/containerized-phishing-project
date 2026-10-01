"""Does the brand a page shows (title / h1 / text) match the domain it is served from?

Split out of the former analyze_dashboard.py in rebuild Phase 2 (code moved unchanged).
"""

from __future__ import annotations
import re
from typing import Any, Dict, List
from phishguard.features.brand_signals import BRAND_TOKENS



def normalize_brand_text(text: str) -> str:
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize_domain_brand(registrable_domain: str) -> List[str]:
    rd = str(registrable_domain or "").lower().strip()
    if not rd:
        return []
    label_raw = rd.split(".", 1)[0]
    label = re.sub(r"[^a-z0-9]", "", label_raw)
    if not label:
        return []
    toks: List[str] = [label]
    for part in re.split(r"[-_]+", label_raw):
        p = normalize_brand_text(part).replace(" ", "")
        if len(p) >= 3:
            toks.append(p)
    if "of" in label and len(label) >= 8:
        for part in re.split(r"of+", label):
            p = normalize_brand_text(part).replace(" ", "")
            if len(p) >= 3:
                toks.append(p)
    for bt in BRAND_TOKENS:
        t = normalize_brand_text(str(bt or "")).replace(" ", "")
        if len(t) >= 4 and t in label:
            toks.append(t)
    seen: set[str] = set()
    out: List[str] = []
    for t in toks:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def extract_primary_brand_candidates(title: str, h1: str, visible_text_sample: str) -> List[str]:
    candidates: List[str] = []
    for raw in (title, h1):
        txt = str(raw or "").strip()
        if not txt:
            continue
        for seg in re.split(r"[|\-:/]", txt):
            n = normalize_brand_text(seg)
            if n:
                candidates.append(" ".join(n.split()[:8]))
    vis = normalize_brand_text(str(visible_text_sample or ""))
    if vis:
        candidates.append(" ".join(vis.split()[:12]))
    seen: set[str] = set()
    out: List[str] = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out[:8]


def compute_brand_domain_coherence(
    *,
    registrable_domain: str,
    title: str,
    h1: str,
    visible_text_sample: str,
) -> Dict[str, Any]:
    domain_tokens = tokenize_domain_brand(registrable_domain)
    page_candidates = extract_primary_brand_candidates(title, h1, visible_text_sample)
    title_n = normalize_brand_text(title)
    h1_n = normalize_brand_text(h1)
    vis_n = normalize_brand_text(visible_text_sample)
    compact_blob = (title_n + " " + h1_n + " " + " ".join(page_candidates)).replace(" ", "")
    score = 0.0
    reasons: List[str] = []
    joined = domain_tokens[0] if domain_tokens else ""
    if len(joined) >= 5 and joined in compact_blob:
        score += 0.65
        reasons.append("joined_domain_token_found_in_title_or_header")
    for tok in domain_tokens[1:] if len(domain_tokens) > 1 else []:
        if len(tok) < 4:
            continue
        if re.search(rf"\b{re.escape(tok)}\b", title_n) or re.search(rf"\b{re.escape(tok)}\b", h1_n):
            score += 0.18
    if joined and len(joined) >= 5 and joined in vis_n.replace(" ", ""):
        score += 0.15
        reasons.append("joined_domain_token_found_in_visible_text")
    match = bool(score >= 0.55)
    reason = "strong_brand_domain_text_alignment" if match else (";".join(reasons) if reasons else "no_strong_alignment")
    return {
        "brand_domain_coherence_score": round(min(score, 1.0), 4),
        "brand_domain_coherence_match": match,
        "brand_domain_coherence_reason": reason,
        "domain_brand_tokens": domain_tokens,
        "page_brand_candidates": page_candidates,
    }
