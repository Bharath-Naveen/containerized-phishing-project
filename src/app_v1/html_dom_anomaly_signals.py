"""HTML/DOM anomaly signals: link/form/resource vs brand and page-host alignment.

Local-only; produces a compact auditable summary. No raw HTML is sent to external APIs.

Page-family heuristics dampen signals on content-heavy / feed-like pages so outbound links,
third-party assets, and incidental brand mentions in article body do not mimic credential
impersonation — without site-specific allowlists.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.pipeline.features.brand_signals import BRAND_TOKENS, host_on_official_brand_apex

from .domain_ecosystem import domain_relation
from .html_structure_signals import _domain_of_action, _host, _uniq_text

_CTA_SUBSTRINGS = (
    "sign in",
    "log in",
    "login",
    "continue",
    "verify",
    "confirm",
    "secure",
    "update",
    "account",
    "help",
    "next",
    "proceed",
    "unlock",
    "restore",
)

_INTERSTITIAL_MARKERS = (
    "continue to",
    "redirecting",
    "you will be redirected",
    "please wait",
    "destination",
    "gateway",
    "preview",
    "you are leaving",
    "external site",
    "transfer you",
)

_COUNTDOWN_MARKERS = (
    "seconds",
    "countdown",
    "redirect in",
    "automatically redirect",
    "auto redirect",
)

_CONTINUE_DEST = ("continue to", "click to continue", "proceed to", "go to")

_DEST_PREVIEW = ("preview", "safe link", "link preview", "you will visit", "final destination")


def _published_time_hint(low_body: str) -> bool:
    """Cheap hint for news/article pages when <time> is not yet counted."""
    return "published" in low_body[:1200] or "posted on" in low_body[:1200]


# Generic discussion / feed cues (no site-specific names).
_FORUM_FEED_MARKERS = (
    "replies",
    "reply to",
    "thread",
    "discussion",
    "comments",
    "commented",
    "posted by",
    "submitted by",
    "permalink",
    "pagination",
    "load more",
    "show more comments",
    "community",
    "forum",
    "moderator",
    "upvote",
    "downvote",
    "subscribe",
    "followers",
)

_STRONG_TRUST_SUBSTRINGS = (
    "sign in",
    "log in",
    "login",
    "sign-in",
    "verify your",
    "verify account",
    "confirm your",
    "unlock",
    "restore account",
    "password",
    "authenticate",
    "2-step",
    "two-step",
    "otp",
)

_WEAK_STANDALONE_CTA = frozenset(
    {
        "continue",
        "next",
        "more",
        "read more",
        "see more",
        "show more",
        "click here",
        "here",
        "go",
        "skip",
        "close",
        "back",
        "»",
        "→",
    }
)

_CONTENT_FAMILY = frozenset({"article_news", "content_feed_forum_aggregator"})


def _text_has_cta(t: str) -> bool:
    tl = (t or "").lower()
    return any(p in tl for p in _CTA_SUBSTRINGS)


def _brands_in_text(t: str) -> Set[str]:
    tl = (t or "").lower()
    return {b for b in BRAND_TOKENS if b in tl}


def _has_strong_trust_substring(t: str) -> bool:
    tl = (t or "").lower()
    return any(s in tl for s in _STRONG_TRUST_SUBSTRINGS) or bool(_brands_in_text(tl))


def _weak_cta_only(text: str) -> bool:
    """Short generic navigation CTA without trust/credential semantics or brand."""
    t = " ".join((text or "").lower().split())
    if not t or len(t) > 56:
        return False
    if _brands_in_text(t):
        return False
    if _has_strong_trust_substring(t):
        return False
    if t in _WEAK_STANDALONE_CTA:
        return True
    parts = t.split()
    if len(parts) <= 4 and parts[0] == "continue" and not any(x in t for x in ("sign", "verify", "account", "password")):
        return True
    return False


def _path_low(url: str) -> str:
    try:
        return (urlparse(url or "").path or "").lower()
    except Exception:
        return ""


def _classify_page_family(
    *,
    base: str,
    low_body: str,
    word_count: int,
    total_links: int,
    external_link_count: int,
    list_item_count: int,
    image_count: int,
    article_tag_count: int,
    p_tag_count: int,
    password_count: int,
    form_count: int,
    interstitial_or_preview_pattern: bool,
    wrapper_page_pattern: bool,
) -> str:
    path = _path_low(base)
    scores: Dict[str, int] = {
        "wrapper_interstitial": 0,
        "auth_login_recovery": 0,
        "checkout_payment": 0,
        "dashboard_admin": 0,
        "article_news": 0,
        "content_feed_forum_aggregator": 0,
        "generic_landing": 1,
    }

    if interstitial_or_preview_pattern or wrapper_page_pattern:
        scores["wrapper_interstitial"] += 95

    if password_count > 0:
        scores["auth_login_recovery"] += 70
    if any(
        x in path
        for x in (
            "/login",
            "/signin",
            "/sign-in",
            "/auth",
            "/session",
            "/recover",
            "/reset",
            "/password",
            "/account/login",
        )
    ):
        scores["auth_login_recovery"] += 35
    if form_count > 0 and password_count == 0:
        if any(x in low_body for x in ("forgot password", "sign in", "log in", "create account")):
            scores["auth_login_recovery"] += 25

    if any(x in path for x in ("/checkout", "/cart", "/pay", "/payment", "/billing", "/order")):
        scores["checkout_payment"] += 55
    if any(x in low_body[:2500] for x in ("credit card", "cvv", "card number", "billing address")):
        scores["checkout_payment"] += 25

    if any(x in path for x in ("/admin", "/dashboard", "/console", "/portal", "/cpanel")):
        scores["dashboard_admin"] += 50

    if article_tag_count >= 1 or (p_tag_count >= 8 and word_count >= 320):
        scores["article_news"] += 45
    if _published_time_hint(low_body):
        scores["article_news"] += 12

    ext_ratio = external_link_count / max(total_links, 1)
    forum_hits = sum(1 for m in _FORUM_FEED_MARKERS if m in low_body)
    if forum_hits >= 2:
        scores["content_feed_forum_aggregator"] += 40
    if total_links >= 38:
        scores["content_feed_forum_aggregator"] += 35
    if total_links >= 22 and ext_ratio >= 0.32:
        scores["content_feed_forum_aggregator"] += 28
    if list_item_count >= 24 and total_links >= 18:
        scores["content_feed_forum_aggregator"] += 22
    if image_count >= 18 and total_links >= 25 and word_count >= 280:
        scores["content_feed_forum_aggregator"] += 18

    if scores["article_news"] >= 40 and scores["content_feed_forum_aggregator"] >= 40:
        scores["content_feed_forum_aggregator"] += 8

    order = (
        "wrapper_interstitial",
        "auth_login_recovery",
        "checkout_payment",
        "dashboard_admin",
        "article_news",
        "content_feed_forum_aggregator",
        "generic_landing",
    )
    best = max(scores.values())
    for name in order:
        if scores[name] == best:
            return name
    return "generic_landing"


def _absolute_url(href: str, base: str) -> str:
    if not href or href.strip().startswith(("#", "javascript:", "mailto:", "tel:")):
        return ""
    try:
        return urljoin(base or "", href.strip())
    except Exception:
        return ""


def _netloc(url: str) -> str:
    if not url:
        return ""
    try:
        return (urlparse(url).netloc or "").lower().split("@")[-1]
    except Exception:
        return ""


def _resource_urls(soup: BeautifulSoup, base: str) -> List[Tuple[str, str]]:
    """Pairs (kind, absolute_url) for scripts, stylesheets, images."""
    out: List[Tuple[str, str]] = []
    for tag in soup.find_all("script", src=True):
        u = _absolute_url(str(tag.get("src") or ""), base)
        if u:
            out.append(("script", u))
    for tag in soup.find_all("link", href=True):
        rel = " ".join(tag.get("rel") or []).lower()
        if "stylesheet" in rel or "icon" in rel:
            u = _absolute_url(str(tag.get("href") or ""), base)
            if u:
                out.append(("link", u))
    for tag in soup.find_all("img", src=True):
        u = _absolute_url(str(tag.get("src") or ""), base)
        if u:
            out.append(("img", u))
    return out


def extract_html_dom_anomaly_signals(
    *,
    html_path: Optional[str],
    final_url: str = "",
    input_url: str = "",
    title_hint: str = "",
    visible_text_hint: str = "",
    soup: Any = None,
) -> Dict[str, Any]:
    """Return DOM anomaly summary, interpretable risk score, and reasons."""
    base = (final_url or input_url or "").strip()

    def _empty(err: str) -> Dict[str, Any]:
        return {
            "html_dom_anomaly_summary": None,
            "html_dom_anomaly_risk_score": None,
            "html_dom_anomaly_reasons": [],
            "html_dom_visual_assessment": "inconclusive",
            "html_dom_anomaly_error": err,
        }

    if soup is None:
        if not html_path:
            return _empty("missing_html_path")
        p = Path(str(html_path))
        if not p.is_file():
            return _empty("html_path_not_found")
        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            return _empty(f"html_read_error:{e}")
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:  # noqa: BLE001
            return _empty(f"html_parse_error:{e}")
    elif not isinstance(soup, BeautifulSoup):
        return _empty("invalid_soup_object")

    final_host = _host(base)
    page_official = bool(final_host and host_on_official_brand_apex(final_host))

    page_title = (soup.title.get_text(" ", strip=True) if soup.title else "") or (title_hint or "")
    h1_el = soup.find("h1")
    h1_text = (h1_el.get_text(" ", strip=True) if h1_el else "")[:200]
    body_text = soup.get_text(" ", strip=True)
    low_body = body_text.lower()
    low_title = page_title.lower()
    low_h1 = h1_text.lower()

    brands_title = _brands_in_text(low_title)
    brands_h1 = _brands_in_text(low_h1)
    brands_body = _brands_in_text(low_body[:4000])
    brands_combined = brands_title | brands_h1 | brands_body

    title_brand_domain_mismatch = bool(
        brands_title and final_host and not page_official and not any(b in final_host for b in brands_title)
    )
    h1_brand_domain_mismatch = bool(
        brands_h1 and final_host and not page_official and not any(b in final_host for b in brands_h1)
    )
    body_brand_domain_mismatch = bool(
        brands_body and final_host and not page_official and not any(b in final_host for b in brands_body)
    )

    strong_branding_without_official_domain = bool(brands_combined and not page_official)

    links = soup.find_all("a")
    total_links = len(links)
    external_link_count = 0
    truly_external_link_count = 0
    for a in links:
        href = str(a.get("href") or "")
        abs_u = _absolute_url(href, base)
        dom = _netloc(abs_u) or _domain_of_action(href, base)
        if dom and final_host and dom != final_host:
            external_link_count += 1
            if domain_relation(final_host, dom)["truly_external"]:
                truly_external_link_count += 1

    list_item_count = len(soup.find_all("li"))
    image_count = len(soup.find_all("img"))
    article_tag_count = len(soup.find_all("article"))
    p_tag_count = len(soup.find_all("p"))
    word_count = len(body_text.split())

    interstitial_or_preview_pattern = any(m in low_body for m in _INTERSTITIAL_MARKERS)
    redirect_countdown_phrase_present = any(m in low_body for m in _COUNTDOWN_MARKERS)
    continue_to_destination_phrase_present = any(m in low_body for m in _CONTINUE_DEST)
    destination_preview_phrase_present = any(m in low_body for m in _DEST_PREVIEW)

    nav = soup.find("nav")
    footer = soup.find("footer")
    nav_links = len(nav.find_all("a")) if nav else 0
    footer_links = len(footer.find_all("a")) if footer else 0
    iframe_count = len(soup.find_all("iframe"))

    explicit_interstitial_text = bool(
        redirect_countdown_phrase_present
        or continue_to_destination_phrase_present
        or destination_preview_phrase_present
    )
    forwarding_heavy = bool(truly_external_link_count >= 2 and total_links <= 8 and word_count < 180)
    wrapper_page_pattern = bool(
        iframe_count >= 1 and word_count < 90 and (explicit_interstitial_text or forwarding_heavy)
    ) or bool(explicit_interstitial_text and forwarding_heavy)

    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    input_types = [str((i.get("type") or "text")).lower() for i in inputs]
    password_count = sum(1 for t in input_types if t == "password")
    email_count = sum(1 for t in input_types if t == "email")

    form_targets_detail: List[Dict[str, Any]] = []
    external_form_domains: Set[str] = set()

    for f in forms:
        act = str(f.get("action") or "").strip()
        dom = _domain_of_action(act, base)
        same = not dom or (bool(final_host) and dom == final_host)
        if dom and final_host and dom != final_host:
            external_form_domains.add(dom)
        form_targets_detail.append(
            {
                "action_domain": dom or "(same-document)",
                "same_host_as_page": bool(same),
            }
        )

    if final_host:
        true_external_form_domains = {
            d for d in external_form_domains if domain_relation(final_host, d)["truly_external"]
        }
    else:
        true_external_form_domains = set(external_form_domains)
    form_action_external_domain_count = len(true_external_form_domains)
    form_action_official_domain_match = form_action_external_domain_count == 0

    actions_list = [str(f.get("action") or "").strip() for f in forms]
    empty_or_hash_only = bool(forms) and all(a == "" or a == "#" for a in actions_list)
    js_submit_only_login_pattern = bool(password_count > 0 and empty_or_hash_only and forms)

    suspicious_credential_collection_pattern = bool(
        password_count > 0
        and (email_count > 0 or any(t in {"text", "email"} for t in input_types))
        and (form_action_external_domain_count > 0 or empty_or_hash_only)
    )

    page_family = _classify_page_family(
        base=base,
        low_body=low_body,
        word_count=word_count,
        total_links=total_links,
        external_link_count=external_link_count,
        list_item_count=list_item_count,
        image_count=image_count,
        article_tag_count=article_tag_count,
        p_tag_count=p_tag_count,
        password_count=password_count,
        form_count=len(forms),
        interstitial_or_preview_pattern=interstitial_or_preview_pattern,
        wrapper_page_pattern=wrapper_page_pattern,
    )

    trust_action_context = bool(
        password_count > 0
        or (email_count > 0 and len(forms) > 0)
        or form_action_external_domain_count > 0
        or js_submit_only_login_pattern
        or suspicious_credential_collection_pattern
        or interstitial_or_preview_pattern
        or wrapper_page_pattern
    )

    content_rich_signals = 0
    if total_links >= 28:
        content_rich_signals += 1
    if total_links >= 45:
        content_rich_signals += 1
    if image_count >= 14:
        content_rich_signals += 1
    if word_count >= 380:
        content_rich_signals += 1
    if truly_external_link_count >= 12 and (truly_external_link_count / max(total_links, 1)) >= 0.22:
        content_rich_signals += 1
    if list_item_count >= 18:
        content_rich_signals += 1

    credential_capture_signals = bool(
        password_count > 0
        or form_action_external_domain_count > 0
        or suspicious_credential_collection_pattern
        or js_submit_only_login_pattern
    )

    content_rich_profile = bool(
        content_rich_signals >= 3
        and not credential_capture_signals
        and page_family in _CONTENT_FAMILY | {"generic_landing"}
    )

    use_strict_anchors = bool(page_family in _CONTENT_FAMILY and not trust_action_context)

    suspicious_anchor_mismatches: List[str] = []
    anchor_mismatch_count = 0
    anchor_strong_mismatch_count = 0
    branded_non_official = 0
    cta_external = 0
    generic_cta_external_link_count = 0
    top_anchor_pairs: List[Dict[str, str]] = []
    link_domain_counter: Counter[str] = Counter()

    seen_pair: Set[str] = set()

    def _record_anchor(text: str, href: str, source: str) -> None:
        nonlocal anchor_mismatch_count, anchor_strong_mismatch_count, branded_non_official, cta_external, generic_cta_external_link_count
        t = " ".join((text or "").split())[:120]
        abs_u = _absolute_url(href, base)
        dom = _netloc(abs_u) or _domain_of_action(href, base)
        if not dom:
            return
        link_domain_counter[dom] += 1
        tl = t.lower()
        brands_here = _brands_in_text(tl)
        cta = _text_has_cta(tl)
        external = bool(final_host and dom and dom != final_host)
        official_target = host_on_official_brand_apex(dom)
        rel = domain_relation(final_host, dom) if (final_host and dom) else {"truly_external": True}
        truly_external = bool(rel["truly_external"])

        strong_trust_text = _has_strong_trust_substring(t)
        weak_only = _weak_cta_only(t)

        if external:
            key = f"{t}|{dom}"
            if key not in seen_pair and len(top_anchor_pairs) < 6:
                seen_pair.add(key)
                top_anchor_pairs.append({"anchor_text": t[:80], "target_domain": dom})

        if not external or official_target or not truly_external:
            return

        suspicious = False

        if use_strict_anchors:
            suspicious = bool(brands_here or strong_trust_text)
            if weak_only and cta:
                generic_cta_external_link_count += 1
            if suspicious and brands_here:
                branded_non_official += 1
        else:
            if brands_here:
                suspicious = True
                branded_non_official += 1
            elif cta:
                if weak_only:
                    generic_cta_external_link_count += 1
                else:
                    suspicious = True
                    cta_external += 1

        if suspicious:
            anchor_mismatch_count += 1
            anchor_strong_mismatch_count += 1
            snippet = f"{source}:{t[:56]!r}→{dom}"
            if len(suspicious_anchor_mismatches) < 8:
                suspicious_anchor_mismatches.append(snippet)

    for a in soup.find_all("a", href=True):
        _record_anchor(a.get_text(" ", strip=True), str(a.get("href") or ""), "a")

    for ar in soup.find_all("area", href=True):
        _record_anchor(ar.get("alt") or ar.get_text(" ", strip=True), str(ar.get("href") or ""), "area")

    for b in soup.find_all("button"):
        fa = str(b.get("formaction") or "").strip()
        if fa:
            _record_anchor(b.get_text(" ", strip=True), fa, "button")

    link_domains_summary = [d for d, _ in link_domain_counter.most_common(12)]

    sparse_credential_capture_layout = bool(
        password_count > 0 and word_count < 220 and total_links <= 14
    )
    missing_real_ecosystem_context = bool(
        nav_links == 0
        and footer_links == 0
        and not any(
            x in low_body
            for x in ("help center", "contact us", "privacy policy", "terms of service", "support")
        )
    )

    login_harvester_pattern = bool(
        password_count > 0
        and sparse_credential_capture_layout
        and (brands_combined or interstitial_or_preview_pattern)
        and not page_official
    )

    suspicious_minimal_login_clone_pattern = bool(
        len(forms) == 1
        and password_count > 0
        and total_links <= 12
        and nav_links == 0
        and footer_links == 0
    )

    label_texts = _uniq_text(
        [lb.get_text(" ", strip=True) for lb in soup.find_all("label")],
        max_items=10,
        max_len=72,
    )
    button_texts_login = _uniq_text(
        [b.get_text(" ", strip=True) for b in soup.find_all("button")]
        + [str(i.get("value") or "") for i in inputs if (i.get("type") or "").lower() == "submit"],
        max_items=10,
        max_len=64,
    )
    trust_surface = " ".join(label_texts) + " " + " ".join(button_texts_login)
    brands_trust_surface = _brands_in_text(trust_surface.lower())
    trust_surface_brand_domain_mismatch = bool(
        brands_trust_surface
        and password_count > 0
        and final_host
        and not page_official
        and not any(b in final_host for b in brands_trust_surface)
    )

    res_pairs = _resource_urls(soup, base)
    res_counter: Counter[str] = Counter()
    for _, u in res_pairs:
        d = _netloc(u)
        if d:
            res_counter[d] += 1
    resource_domains_summary = [d for d, _ in res_counter.most_common(14)]

    branded_res_non_official = 0
    logo_domain_mismatch = False
    for kind, u in res_pairs:
        lu = u.lower()
        brands_in_url = _brands_in_text(lu)
        d = _netloc(u)
        if brands_in_url and d and not host_on_official_brand_apex(d):
            branded_res_non_official += 1
        if (
            kind == "img"
            and brands_in_url
            and d
            and final_host
            and d != final_host
            and not host_on_official_brand_apex(d)
        ):
            logo_domain_mismatch = True

    strong_impersonation_context = bool(
        trust_action_context
        or form_action_external_domain_count > 0
        or anchor_strong_mismatch_count > 0
        or trust_surface_brand_domain_mismatch
    )

    risk = 0.0
    reasons: List[str] = []

    eff_anchor_count = anchor_strong_mismatch_count if use_strict_anchors else anchor_mismatch_count
    if eff_anchor_count:
        risk += min(0.28, 0.09 + 0.05 * min(eff_anchor_count, 4))
        reasons.append(
            f"Trust-sensitive anchors point off-host ({eff_anchor_count}); targets are not official brand apex hosts."
        )

    if form_action_external_domain_count:
        risk += min(0.30, 0.12 + 0.06 * form_action_external_domain_count)
        reasons.append("Credential form submits to a different host than the captured page.")

    res_dampen = 1.0
    if page_family in _CONTENT_FAMILY and not trust_action_context:
        res_dampen = 0.28
    elif content_rich_profile and not credential_capture_signals:
        res_dampen = 0.45
    if branded_res_non_official:
        risk += res_dampen * min(0.18, 0.06 + 0.04 * min(branded_res_non_official, 3))
        if res_dampen < 1.0:
            reasons.append(
                "Third-party assets reference brand-like path segments; weighted lightly on content-heavy pages."
            )
        else:
            reasons.append("Scripts, styles, or images reference brand-like paths on non-official hosts.")

    logo_w = 1.0
    if content_rich_profile and image_count >= 20:
        logo_w = 0.35
    if logo_domain_mismatch:
        risk += 0.10 * logo_w
        if logo_w < 1.0:
            reasons.append("Logo/host alignment flagged but discounted on image-rich pages.")
        else:
            reasons.append("Logo or branded image loads from a host that does not match an official brand apex.")

    # Brand-in-text vs host: title/H1 medium; body weak unless trust surface or auth-like page.
    title_w, h1_w, body_w = 1.0, 0.75, 0.55
    if page_family in _CONTENT_FAMILY:
        body_w = 0.08
        title_w = 0.42 if not trust_action_context else 0.85
        h1_w = 0.38 if not trust_action_context else 0.8
    elif page_family == "generic_landing" and not trust_action_context:
        body_w = 0.35
        title_w = 0.55
        h1_w = 0.5

    if not strong_impersonation_context:
        body_w = min(body_w, 0.12)

    brand_text_risk = 0.0
    if title_brand_domain_mismatch:
        brand_text_risk += 0.12 * title_w
    if h1_brand_domain_mismatch:
        brand_text_risk += 0.10 * h1_w
    if body_brand_domain_mismatch:
        brand_text_risk += 0.10 * body_w

    if brand_text_risk > 0:
        risk += brand_text_risk
        reasons.append(
            "Brand-like wording in title/headline/body does not match the page host family (weighted by page type and trust context)."
        )

    if trust_surface_brand_domain_mismatch:
        risk += 0.14
        reasons.append("Password/login labels or submit buttons reference a brand the host does not belong to.")

    if interstitial_or_preview_pattern or wrapper_page_pattern:
        risk += 0.10
        reasons.append("Language or layout suggests a redirect, preview, or wrapper around another destination.")

    if suspicious_minimal_login_clone_pattern or login_harvester_pattern:
        risk += 0.14
        reasons.append("Minimal login-only surface with little surrounding site navigation or help content.")

    if js_submit_only_login_pattern and password_count > 0:
        risk += 0.08
        reasons.append("Password form with empty or fragment action (often JS-driven submit), which is harder to audit.")

    dampener_factor = 1.0
    if content_rich_profile and page_family in _CONTENT_FAMILY:
        dampener_factor = 0.42
    elif content_rich_profile:
        dampener_factor = 0.58
    elif page_family in _CONTENT_FAMILY and not trust_action_context:
        dampener_factor = 0.68

    risk *= dampener_factor
    if dampener_factor < 1.0:
        reasons.append(
            "Content-rich / feed-like profile with no credential capture: DOM anomaly score dampened."
        )

    aligned = page_official and form_action_external_domain_count == 0 and eff_anchor_count <= 1
    if aligned and (nav_links + footer_links) >= 6:
        risk -= 0.14
        reasons.append("Official host with on-page submits and richer nav/footer structure reduces DOM anomaly risk.")

    risk = max(0.0, min(1.0, risk))

    summary = {
        "page_family": page_family,
        "trust_action_context": trust_action_context,
        "strong_impersonation_context": strong_impersonation_context,
        "content_rich_profile": content_rich_profile,
        "content_rich_dampener_factor": round(dampener_factor, 4),
        "content_rich_signal_count": content_rich_signals,
        "same_ecosystem_external_links_suppressed": max(0, external_link_count - truly_external_link_count),
        "anchor_text_to_domain_mismatch_count": anchor_mismatch_count,
        "anchor_strong_mismatch_count": anchor_strong_mismatch_count,
        "generic_cta_external_link_count": generic_cta_external_link_count,
        "strict_anchor_filter_active": use_strict_anchors,
        "suspicious_anchor_target_mismatches": suspicious_anchor_mismatches,
        "branded_anchor_targets_non_official": branded_non_official,
        "cta_links_to_external_domains": cta_external,
        "link_domains_summary": link_domains_summary,
        "top_anchor_texts_with_target_domains": top_anchor_pairs,
        "form_action_targets_summary": form_targets_detail[:8],
        "form_action_official_domain_match": form_action_official_domain_match,
        "form_action_external_domain_count": form_action_external_domain_count,
        "form_action_raw_cross_host_count": len(external_form_domains),
        "js_submit_only_login_pattern": js_submit_only_login_pattern,
        "suspicious_credential_collection_pattern": suspicious_credential_collection_pattern,
        "resource_domains_summary": resource_domains_summary,
        "branded_resource_domains_non_official": branded_res_non_official,
        "title_brand_domain_mismatch": title_brand_domain_mismatch,
        "h1_brand_domain_mismatch": h1_brand_domain_mismatch,
        "body_brand_domain_mismatch": body_brand_domain_mismatch,
        "trust_surface_brand_domain_mismatch": trust_surface_brand_domain_mismatch,
        "logo_domain_mismatch": logo_domain_mismatch,
        "interstitial_or_preview_pattern": interstitial_or_preview_pattern,
        "redirect_countdown_phrase_present": redirect_countdown_phrase_present,
        "continue_to_destination_phrase_present": continue_to_destination_phrase_present,
        "destination_preview_phrase_present": destination_preview_phrase_present,
        "wrapper_page_pattern": wrapper_page_pattern,
        "login_harvester_pattern": login_harvester_pattern,
        "sparse_credential_capture_layout": sparse_credential_capture_layout,
        "missing_real_ecosystem_context": missing_real_ecosystem_context,
        "strong_branding_without_official_domain": strong_branding_without_official_domain,
        "suspicious_minimal_login_clone_pattern": suspicious_minimal_login_clone_pattern,
        "visible_label_texts_sample": label_texts,
        "page_host_official_brand_apex": page_official,
    }

    if interstitial_or_preview_pattern or wrapper_page_pattern:
        assessment = "wrapper_interstitial"
    elif login_harvester_pattern or suspicious_minimal_login_clone_pattern:
        assessment = "credential_harvester_risk"
    elif page_official and form_action_external_domain_count == 0 and eff_anchor_count <= 1 and risk < 0.22:
        assessment = "legitimate_auth_flow_likely"
    elif (
        form_action_external_domain_count > 0
        or trust_surface_brand_domain_mismatch
        or (risk >= 0.32 and strong_impersonation_context)
        or (anchor_strong_mismatch_count >= 2 and strong_impersonation_context)
    ):
        assessment = "suspicious_impersonation"
    else:
        assessment = "inconclusive"

    return {
        "html_dom_anomaly_summary": summary,
        "html_dom_anomaly_risk_score": round(risk, 4),
        "html_dom_anomaly_reasons": reasons,
        "html_dom_visual_assessment": assessment,
        "html_dom_anomaly_error": None,
    }
