"""HTTP fetch + DOM/HTML statistics (passive; no form submission)."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0
_MAX_REDIRECTS = 10
_USER_AGENT = (
    "Mozilla/5.0 (compatible; PhishPipeline/1.0; +https://example.invalid; research)"
)


def _same_registrable(a: str, b: str) -> bool:
    try:
        import tldextract

        ea = tldextract.extract(a)
        eb = tldextract.extract(b)
        ra = ".".join(p for p in (ea.domain, ea.suffix) if p)
        rb = ".".join(p for p in (eb.domain, eb.suffix) if p)
        return ra == rb and ra != ""
    except Exception:
        return a.lower() == b.lower()


def fetch_html_http(url: str) -> Tuple[Dict[str, Any], str]:
    """Return feature dict prefix + raw html string (may be empty)."""
    out: Dict[str, Any] = {}
    html = ""
    headers = {"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    try:
        with httpx.Client(
            follow_redirects=True,
            max_redirects=_MAX_REDIRECTS,
            timeout=_DEFAULT_TIMEOUT,
            headers=headers,
        ) as client:
            resp = client.get(url)
            out["http_status"] = resp.status_code
            chain = [str(r.headers.get("location", r.url)) for r in resp.history] + [
                str(resp.url)
            ]
            # history doesn't include first url; hops = len(history)+1 after final
            initial_host = (urlsplit(url).hostname or "").lower()
            hops = len(resp.history) + 1
            out["redirect_hop_count"] = max(0, hops - 1)
            cross = 0
            prev = initial_host
            for r in resp.history:
                loc = r.headers.get("location")
                if not loc:
                    continue
                next_host = (urlsplit(str(r.url)).hostname or "").lower()
                if prev and next_host and prev != next_host:
                    if not _same_registrable(prev, next_host):
                        cross += 1
                prev = next_host or prev
            final_host = (urlsplit(str(resp.url)).hostname or "").lower()
            if initial_host and final_host and initial_host != final_host:
                if not _same_registrable(initial_host, final_host):
                    cross += 1
            out["cross_domain_redirect_count"] = cross
            out["final_url"] = str(resp.url)
            ct = resp.headers.get("content-type", "")
            if "text/html" in ct or "application/xhtml" in ct or not ct:
                html = resp.text or ""
            else:
                html = ""
            out["page_fetch_success"] = int(resp.status_code < 400 and bool(html))
            if not html:
                out["fetch_error"] = "empty_or_non_html_body"
            else:
                out["fetch_error"] = ""
    except httpx.TimeoutException:
        out["http_status"] = 0
        out["redirect_hop_count"] = 0
        out["cross_domain_redirect_count"] = 0
        out["final_url"] = url
        out["page_fetch_success"] = 0
        out["fetch_error"] = "http_timeout"
    except Exception as e:
        logger.debug("fetch failed: %s", e)
        out["http_status"] = 0
        out["redirect_hop_count"] = 0
        out["cross_domain_redirect_count"] = 0
        out["final_url"] = url
        out["page_fetch_success"] = 0
        out["fetch_error"] = f"http_error:{type(e).__name__}"
    return out, html


def _visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)


def _safe_netloc(u: str) -> str:
    return (urlsplit(u).hostname or "").lower()


def extract_dom_features(
    input_url: str,
    fetch_out: Dict[str, Any],
    html: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    base_final = fetch_out.get("final_url") or input_url
    base_host = _safe_netloc(base_final)
    html = html or ""
    out["html_length"] = len(html)
    if not html.strip():
        out["title_length"] = 0
        out["visible_text_length"] = 0
        out["form_count"] = 0
        out["input_count"] = 0
        out["password_field_count"] = 0
        out["email_field_count"] = 0
        out["hidden_input_count"] = 0
        out["button_count"] = 0
        out["link_count"] = 0
        out["external_link_count"] = 0
        out["iframe_count"] = 0
        out["script_count"] = 0
        out["external_script_count"] = 0
        out["image_count"] = 0
        out["meta_refresh_present"] = 0
        out["favicon_present"] = 0
        out["off_domain_favicon_flag"] = 0
        out["form_action_count"] = 0
        out["off_domain_form_action_count"] = 0
        out["heading_count"] = 0
        out["h1_count"] = 0
        out["h2_count"] = 0
        out["lang_attr"] = ""
        out["html_features_missing"] = 1
        out["_title_text"] = ""
        out["_visible_text"] = ""
        return out

    soup = BeautifulSoup(html, "html.parser")
    out["html_features_missing"] = 0
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    out["title_length"] = len(title)
    vis = _visible_text(soup)
    out["visible_text_length"] = len(vis)

    forms = soup.find_all("form")
    out["form_count"] = len(forms)
    inputs = soup.find_all("input")
    out["input_count"] = len(inputs)
    out["password_field_count"] = sum(
        1 for i in inputs if (i.get("type") or "").lower() == "password"
    )
    out["email_field_count"] = sum(
        1
        for i in inputs
        if (i.get("type") or "").lower() in {"email", "text"}
        and "mail" in (i.get("name") or "" + (i.get("id") or "")).lower()
    )
    out["hidden_input_count"] = sum(
        1 for i in inputs if (i.get("type") or "").lower() == "hidden"
    )
    btn_inputs = sum(
        1
        for i in inputs
        if (i.get("type") or "").lower() in {"submit", "button"}
    )
    out["button_count"] = len(soup.find_all("button")) + btn_inputs
    links = soup.find_all("a", href=True)
    out["link_count"] = len(links)
    ext = 0
    for a in links:
        href = a.get("href") or ""
        abs_u = urljoin(base_final, href)
        h = _safe_netloc(abs_u)
        if h and base_host and h != base_host:
            ext += 1
    out["external_link_count"] = ext

    out["iframe_count"] = len(soup.find_all("iframe"))
    scripts = soup.find_all("script", src=True)
    out["script_count"] = len(soup.find_all("script"))
    exs = 0
    for s in scripts:
        src = s.get("src") or ""
        abs_u = urljoin(base_final, src)
        h = _safe_netloc(abs_u)
        if h and base_host and h != base_host:
            exs += 1
    out["external_script_count"] = exs
    out["image_count"] = len(soup.find_all("img"))

    out["meta_refresh_present"] = int(bool(soup.find("meta", attrs={"http-equiv": re.compile("^refresh$", re.I)})))

    icon_hrefs: List[str] = []
    for link in soup.find_all("link", rel=True):
        rel = " ".join(link.get("rel") or []).lower()
        if "icon" in rel:
            href = link.get("href")
            if href:
                icon_hrefs.append(urljoin(base_final, href))
    out["favicon_present"] = int(len(icon_hrefs) > 0)
    off_fav = 0
    for fh in icon_hrefs:
        if _safe_netloc(fh) and base_host and _safe_netloc(fh) != base_host:
            off_fav = 1
    out["off_domain_favicon_flag"] = off_fav

    fa = 0
    off_fa = 0
    for f in forms:
        action = f.get("action")
        fa += 1
        if action:
            au = urljoin(base_final, action)
            ah = _safe_netloc(au)
            if ah and base_host and ah != base_host:
                off_fa += 1
    out["form_action_count"] = fa
    out["off_domain_form_action_count"] = off_fa

    hs = soup.find_all(re.compile("^h[1-6]$", re.I))
    out["heading_count"] = len(hs)
    out["h1_count"] = len(soup.find_all("h1"))
    out["h2_count"] = len(soup.find_all("h2"))
    html_tag = soup.find("html")
    out["lang_attr"] = (html_tag.get("lang") or "").strip() if html_tag else ""
    out["_title_text"] = title
    out["_visible_text"] = vis[:8000]
    return out
