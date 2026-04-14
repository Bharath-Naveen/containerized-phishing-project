"""Compact HTML structure signal extraction for adjudication/explanations.

This parser is local-only and returns compact summaries; it never sends raw HTML upstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.pipeline.features.brand_signals import BRAND_TOKENS, host_on_official_brand_apex
from src.pipeline.safe_url import safe_hostname

_SUSPICIOUS_PHRASES = (
    "verify account",
    "unusual activity",
    "confirm identity",
    "suspended",
    "security alert",
    "payment failed",
    "limited account",
    "update payment",
    "urgent action",
)


def _uniq_text(items: List[str], max_items: int = 8, max_len: int = 80) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        t = (x or "").strip()
        if not t:
            continue
        t = " ".join(t.split())[:max_len]
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= max_items:
            break
    return out


def _host(url: str) -> str:
    h, _ = safe_hostname(url)
    return h


def _domain_of_action(action: str, final_url: str) -> str:
    if not action:
        return ""
    try:
        absolute = urljoin(final_url, action) if final_url else action
        return (urlparse(absolute).netloc or "").lower()
    except Exception:
        return ""


def extract_html_structure_signals(
    *,
    html_path: Optional[str],
    final_url: str = "",
    input_url: str = "",
    title_hint: str = "",
    visible_text_hint: str = "",
    soup: Any = None,
) -> Dict[str, Any]:
    """Return compact HTML structure summary + optional structural risk score.

    If ``soup`` is provided (e.g. from a shared parse in the dashboard), the file at
    ``html_path`` is not read; ``html_path`` may still be absent when ``soup`` is set.
    """
    if soup is None:
        if not html_path:
            return {
                "html_structure_summary": None,
                "html_structure_risk_score": None,
                "html_structure_reasons": [],
                "html_structure_error": "missing_html_path",
            }
        p = Path(str(html_path))
        if not p.is_file():
            return {
                "html_structure_summary": None,
                "html_structure_risk_score": None,
                "html_structure_reasons": [],
                "html_structure_error": "html_path_not_found",
            }
        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            return {
                "html_structure_summary": None,
                "html_structure_risk_score": None,
                "html_structure_reasons": [],
                "html_structure_error": f"html_read_error:{e}",
            }
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:  # noqa: BLE001
            return {
                "html_structure_summary": None,
                "html_structure_risk_score": None,
                "html_structure_reasons": [],
                "html_structure_error": f"html_parse_error:{e}",
            }
    elif not isinstance(soup, BeautifulSoup):
        return {
            "html_structure_summary": None,
            "html_structure_risk_score": None,
            "html_structure_reasons": [],
            "html_structure_error": "invalid_soup_object",
        }

    page_title = (soup.title.get_text(" ", strip=True) if soup.title else "") or (title_hint or "")
    h1_text = (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "")[:180]
    h2_texts = _uniq_text([x.get_text(" ", strip=True) for x in soup.find_all("h2")], max_items=5, max_len=100)

    body_text = soup.get_text(" ", strip=True)
    low_text = body_text.lower()
    visible_snip = (body_text[:900] if body_text else (visible_text_hint or "")[:900])
    suspicious_hits = [p for p in _SUSPICIOUS_PHRASES if p in low_text][:6]

    brands_in_text = sorted({b for b in BRAND_TOKENS if b in low_text or b in page_title.lower()})[:8]

    forms = soup.find_all("form")
    form_count = len(forms)
    inputs = soup.find_all("input")
    input_type = [str((i.get("type") or "text")).lower() for i in inputs]
    password_count = sum(1 for t in input_type if t == "password")
    email_count = sum(1 for t in input_type if t == "email")
    phone_count = sum(1 for t in input_type if t in {"tel", "phone"})
    hidden_count = sum(1 for t in input_type if t == "hidden")
    submit_count = sum(1 for t in input_type if t in {"submit", "button"})
    submit_count += len([b for b in soup.find_all("button") if (b.get("type") or "").lower() in {"", "submit"}])

    button_texts = _uniq_text(
        [b.get_text(" ", strip=True) for b in soup.find_all("button")]
        + [str(i.get("value") or "") for i in inputs if (i.get("type") or "").lower() == "submit"],
        max_items=8,
        max_len=64,
    )
    input_names = _uniq_text([str(i.get("name") or "") for i in inputs], max_items=10, max_len=40)
    placeholders = _uniq_text([str(i.get("placeholder") or "") for i in inputs], max_items=10, max_len=64)

    form_actions = [str(f.get("action") or "").strip() for f in forms]
    form_domains = _uniq_text([_domain_of_action(a, final_url or input_url) for a in form_actions], max_items=8, max_len=80)
    final_host = _host(final_url or input_url)
    cross_domain_action = any(d and final_host and d != final_host for d in form_domains)
    empty_action = any((a == "" or a == "#") for a in form_actions) if forms else False
    suspicious_action_hint = bool(cross_domain_action or (password_count > 0 and empty_action))

    links = soup.find_all("a")
    total_links = len(links)
    nav = soup.find("nav")
    footer = soup.find("footer")
    nav_links = len(nav.find_all("a")) if nav else 0
    footer_links = len(footer.find_all("a")) if footer else 0
    external_links = 0
    for a in links:
        href = str(a.get("href") or "")
        dom = _domain_of_action(href, final_url or input_url)
        if dom and final_host and dom != final_host:
            external_links += 1

    iframe_count = len(soup.find_all("iframe"))
    script_count = len(soup.find_all("script"))
    image_count = len(soup.find_all("img"))

    single_primary_form = bool(form_count == 1 and password_count >= 1)
    sparse_login_layout = bool(single_primary_form and total_links <= 6 and nav_links == 0 and footer_links == 0)
    missing_nav = nav_links == 0
    missing_footer = footer_links == 0

    host_official = bool(final_host and host_on_official_brand_apex(final_host))
    brands_not_in_domain = [b for b in brands_in_text if b not in final_host][:8]
    logo_brands = sorted(
        {
            b
            for img in soup.find_all("img")
            for b in BRAND_TOKENS
            if b in (str(img.get("src") or "").lower())
        }
    )[:8]
    html_brand_mismatch = bool(brands_not_in_domain and not host_official)

    lower = low_text
    has_captcha = ("captcha" in lower) or (soup.find(attrs={"id": lambda x: isinstance(x, str) and "captcha" in x.lower()}) is not None)
    has_remember_me = ("remember me" in lower) or any("remember" in t.lower() for t in button_texts)
    has_forgot_password = ("forgot password" in lower) or ("password reset" in lower)
    has_create_account = ("create account" in lower) or ("sign up" in lower)
    has_support_help = ("support" in lower) or ("help center" in lower) or ("contact us" in lower)

    risk = 0.0
    reasons: List[str] = []
    if cross_domain_action:
        risk += 0.30
        reasons.append("Form action posts to a different domain than the captured page.")
    if sparse_login_layout and password_count > 0:
        risk += 0.22
        reasons.append("Sparse single-form login layout with weak surrounding navigation/help structure.")
    if html_brand_mismatch:
        risk += 0.24
        reasons.append("Brand terms appear in page content but domain does not match an official family.")
    if suspicious_hits:
        risk += 0.10
        reasons.append("Sensitive urgency/security phrases found in page text.")
    if host_official and (nav_links + footer_links) >= 8 and has_support_help:
        risk -= 0.12
        reasons.append("Rich official-site structure signals (nav/footer/help links) reduce HTML risk.")
    risk = max(0.0, min(1.0, risk))

    summary = {
        "title": page_title[:200],
        "h1_text": h1_text,
        "h2_texts": h2_texts,
        "visible_text_snippet": visible_snip,
        "suspicious_phrase_hits": suspicious_hits,
        "brand_terms_found_in_text": brands_in_text,
        "form_count": form_count,
        "password_input_count": password_count,
        "email_input_count": email_count,
        "phone_input_count": phone_count,
        "hidden_input_count": hidden_count,
        "submit_button_count": submit_count,
        "button_texts": button_texts,
        "input_names_summary": input_names,
        "placeholders_summary": placeholders,
        "form_action_urls": _uniq_text(form_actions, max_items=6, max_len=120),
        "form_action_domains": form_domains,
        "cross_domain_form_action": bool(cross_domain_action),
        "empty_or_missing_form_action": bool(empty_action),
        "suspicious_form_action_hint": bool(suspicious_action_hint),
        "total_link_count": total_links,
        "nav_link_count": nav_links,
        "footer_link_count": footer_links,
        "external_link_count": external_links,
        "iframe_count": iframe_count,
        "script_count": script_count,
        "image_count": image_count,
        "page_has_single_primary_form": single_primary_form,
        "sparse_login_like_layout": sparse_login_layout,
        "missing_navigation_structure": missing_nav,
        "missing_footer_structure": missing_footer,
        "brand_terms_not_in_domain": brands_not_in_domain,
        "logo_or_img_src_brand_terms": logo_brands,
        "official_domain_family_match": host_official,
        "brand_mismatch_from_html_context": html_brand_mismatch,
        "has_captcha": has_captcha,
        "has_remember_me_checkbox": has_remember_me,
        "has_forgot_password_link": has_forgot_password,
        "has_create_account_link": has_create_account,
        "has_support_help_links": has_support_help,
    }
    return {
        "html_structure_summary": summary,
        "html_structure_risk_score": round(risk, 4),
        "html_structure_reasons": reasons,
        "html_structure_error": None,
    }
