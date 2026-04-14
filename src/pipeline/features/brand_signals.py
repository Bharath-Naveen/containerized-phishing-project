"""Structural brand / impersonation features (learnable signals, not verdict rules).

Design:
- **Anchors**: registrable domain (eTLD+1) in a moderate curated set of known corporate roots.
  This is a *feature*, not an allowlist classifier — the model combines it with everything else.
- **Mismatch**: brand tokens in path, hyphenated fake labels (``google-login``), typosquat-like
  embedding, brand cues on free-hosting registrable domains, etc.

Whole-label matching reduces false positives on ``accounts.google.com`` (label ``google``) vs
substring hits on ``google.com`` (still label ``google``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

# Tokens commonly abused in phishing (keep compact; model generalizes to similar patterns).
BRAND_TOKENS: tuple[str, ...] = (
    "amazon",
    "apple",
    "facebook",
    "google",
    "instagram",
    "linkedin",
    "microsoft",
    "microsoftonline",
    "netflix",
    "paypal",
    "whatsapp",
    "icloud",
    "outlook",
    "meta",
)

# Known-legitimate registrable domains (eTLD+1), lowercase. Curated superset across the tokens above.
# Subdomains (e.g. accounts.google.com) collapse to these via tldextract.
# eTLD+1 domain labels (leftmost label of registrable) that are legitimate corporate roots.
_OFFICIAL_DOMAIN_LABELS: frozenset[str] = frozenset(
    {
        "google",
        "amazon",
        "apple",
        "facebook",
        "paypal",
        "netflix",
        "microsoftonline",
        "live",
        "office",
        "outlook",
        "icloud",
        "instagram",
        "whatsapp",
        "linkedin",
        "microsoft",
        "meta",
        "fb",
        "okta",
        "auth0",
        "duosecurity",
        "cloudflare",
        "github",
        "gitlab",
        "stripe",
        "shopify",
        "atlassian",
        "slack",
        "zoom",
        "box",
        "docusign",
        "dropbox",
        "salesforce",
    }
)


OFFICIAL_REGISTERED_ANCHORS: frozenset[str] = frozenset(
    {
        "amazon.com",
        "amazon.co.uk",
        "amazon.de",
        "amazon.fr",
        "amazon.ca",
        "amazon.in",
        "amazon.com.au",
        "amazonaws.com",
        "apple.com",
        "icloud.com",
        "facebook.com",
        "fb.com",
        "meta.com",
        "google.com",
        "google.de",
        "google.co.uk",
        "google.fr",
        "microsoft.com",
        "microsoftonline.com",
        "live.com",
        "office.com",
        "outlook.com",
        "windows.net",
        "paypal.com",
        "netflix.com",
        "linkedin.com",
        "instagram.com",
        "whatsapp.com",
        "okta.com",
        "auth0.com",
        "duosecurity.com",
        "cloudflare.com",
        "github.com",
        "gitlab.com",
        "stripe.com",
        "shopify.com",
        "atlassian.net",
        "slack.com",
        "zoom.us",
        "box.com",
        "docusign.com",
        "dropbox.com",
        "salesforce.com",
    }
)


def _host_labels(host: str) -> list[str]:
    h = (host or "").lower().strip(".").split(":")[0]
    if not h or h.startswith("["):
        return []
    return [p for p in h.split(".") if p]


def _labels_matching_brands(labels: Sequence[str]) -> Set[str]:
    found: Set[str] = set()
    for lab in labels:
        for b in BRAND_TOKENS:
            if lab == b:
                found.add(b)
    return found


def _hyphenated_brand_prefix_deception(labels: Sequence[str]) -> int:
    """Label starts with ``brand-`` and continues (e.g. google-login-secure)."""
    for lab in labels:
        for b in BRAND_TOKENS:
            if lab.startswith(b + "-") and len(lab) > len(b) + 1:
                return 1
    return 0


def _typosquat_embedded_in_label(labels: Sequence[str]) -> int:
    """Brand prefix glued to junk (``googleevil``) — skips known official domain labels."""
    for lab in labels:
        if lab in _OFFICIAL_DOMAIN_LABELS:
            continue
        for b in BRAND_TOKENS:
            if lab == b or lab.startswith(b + "-"):
                continue
            if lab.startswith(b) and len(lab) > len(b):
                return 1
    return 0


def _brand_substrings_in(s: str) -> Set[str]:
    sl = (s or "").lower()
    return {b for b in BRAND_TOKENS if b in sl}


def extract_brand_structure_features(
    *,
    host: str,
    path: str,
    registered_domain: str,
    free_hosting_flag: int,
    cloud_hosting_flag: int,
) -> Dict[str, Any]:
    reg = (registered_domain or "").lower().strip()
    pl = (path or "").lower()

    labels = _host_labels(host)
    exact_brands_in_labels = _labels_matching_brands(labels)

    official_registrable_anchor = int(reg in OFFICIAL_REGISTERED_ANCHORS if reg else 0)
    reg_root_label = reg.split(".")[0] if reg and "." in reg else (reg or "")
    official_domain_family = int(
        official_registrable_anchor == 1
        or (
            bool(reg_root_label)
            and reg_root_label in _OFFICIAL_DOMAIN_LABELS
            and len(reg.split(".")) >= 2
        )
    )
    brand_hostname_exact_label_match = int(len(exact_brands_in_labels) > 0)
    brand_hyphenated_deception_label = _hyphenated_brand_prefix_deception(labels)
    brand_typosquat_embedded_in_label = _typosquat_embedded_in_label(labels)

    brands_in_path = _brand_substrings_in(pl)
    path_brand_token_present = int(len(brands_in_path) > 0)

    path_brand_without_official_host = int(
        path_brand_token_present == 1
        and official_registrable_anchor == 0
        and brand_hostname_exact_label_match == 0
    )

    host_lower = (host or "").lower()
    brands_in_host_sub = _brand_substrings_in(host_lower)

    suspicious_host_brand_cue = int(
        len(brands_in_host_sub) > 0
        and official_registrable_anchor == 0
        and brand_hostname_exact_label_match == 0
        and brand_hyphenated_deception_label == 0
    )

    brand_on_free_hosting = int(
        free_hosting_flag == 1
        and (
            path_brand_token_present
            or brand_hyphenated_deception_label
            or brand_typosquat_embedded_in_label
            or suspicious_host_brand_cue
        )
    )
    brand_on_cloud_placeholder = int(
        cloud_hosting_flag == 1
        and (
            path_brand_token_present
            or brand_hyphenated_deception_label
            or brand_typosquat_embedded_in_label
            or suspicious_host_brand_cue
        )
    )

    host_brand_substring_not_official = int(
        len(brands_in_host_sub) > 0 and official_registrable_anchor == 0
    )

    n_brand_hints_host = min(3, len(brands_in_host_sub))
    n_brand_hints_path = min(3, len(brands_in_path))

    return {
        "official_registrable_anchor": official_registrable_anchor,
        "official_domain_family": official_domain_family,
        "brand_hostname_exact_label_match": brand_hostname_exact_label_match,
        "brand_hyphenated_deception_label": brand_hyphenated_deception_label,
        "brand_typosquat_embedded_in_label": brand_typosquat_embedded_in_label,
        "path_brand_token_present": path_brand_token_present,
        "path_brand_without_official_host": path_brand_without_official_host,
        "brand_on_free_hosting": brand_on_free_hosting,
        "brand_on_cloud_placeholder": brand_on_cloud_placeholder,
        "host_brand_substring_not_official": host_brand_substring_not_official,
        "num_brand_tokens_in_host": n_brand_hints_host,
        "num_brand_tokens_in_path": n_brand_hints_path,
    }


BRAND_STRUCTURE_FEATURE_KEYS: tuple[str, ...] = (
    "official_registrable_anchor",
    "official_domain_family",
    "brand_hostname_exact_label_match",
    "brand_hyphenated_deception_label",
    "brand_typosquat_embedded_in_label",
    "path_brand_token_present",
    "path_brand_without_official_host",
    "brand_on_free_hosting",
    "brand_on_cloud_placeholder",
    "host_brand_substring_not_official",
    "num_brand_tokens_in_host",
    "num_brand_tokens_in_path",
    "layer1_brand_trust_score",
)


def explain_brand_structure_features(feat: Dict[str, Any]) -> List[str]:
    """Human-readable lines for dashboard / CLI (derived from computed features)."""
    lines: List[str] = []
    if int(feat.get("official_registrable_anchor") or 0):
        lines.append("The registrable domain matches a known official corporate domain family (anchor feature).")
    elif int(feat.get("official_domain_family") or 0):
        lines.append("The registrable domain’s primary label matches a known corporate / SaaS brand family (legitimacy-anchor feature; not a full registrable-anchor match).")
    if int(feat.get("brand_hostname_exact_label_match") or 0):
        lines.append("A tracked brand name appears as its own hostname label (consistent with legitimate subdomains such as accounts.<brand>.com).")
    if int(feat.get("layer1_brand_trust_score") or 0) >= 3:
        lines.append("Combined official-registrable + brand-label signals indicate a high-trust host pattern (learned feature).")
    if int(feat.get("brand_hyphenated_deception_label") or 0):
        lines.append("A hostname label uses a hyphenated brand prefix pattern often seen in impersonation (e.g. brand-words-other).")
    if int(feat.get("brand_typosquat_embedded_in_label") or 0):
        lines.append("A brand token is embedded inside a longer hostname label (possible typosquat / lookalike).")
    if int(feat.get("path_brand_without_official_host") or 0):
        lines.append("A brand token appears in the path but the host does not look like an official registrable domain for that brand.")
    if int(feat.get("brand_on_free_hosting") or 0):
        lines.append("Brand-related cues appear on a free-hosting or consumer-site registrable domain.")
    if int(feat.get("brand_on_cloud_placeholder") or 0):
        lines.append("Brand-related cues appear together with cloud / PaaS-style hosting patterns.")
    if int(feat.get("host_brand_substring_not_official") or 0) and not int(feat.get("official_registrable_anchor") or 0):
        if not int(feat.get("brand_hostname_exact_label_match") or 0):
            lines.append("Brand-like text appears in the hostname, but the registrable domain is not in the official-anchor feature set.")
    return lines


# Backwards-compatible name for dashboard cap / older imports
def host_on_official_brand_apex(host: str) -> bool:
    """True if hostname is a subdomain of an official registrable anchor (cheap structural check)."""
    h = (host or "").lower().strip(".").split(":")[0]
    if not h:
        return False
    ext_reg = None
    try:
        import tldextract

        ext = tldextract.extract(h)
        ext_reg = ".".join(p for p in (ext.domain, ext.suffix) if p).lower()
    except Exception:
        ext_reg = None
    if ext_reg and ext_reg in OFFICIAL_REGISTERED_ANCHORS:
        return True
    for apex in OFFICIAL_REGISTERED_ANCHORS:
        if h == apex or h.endswith("." + apex):
            return True
    return False
