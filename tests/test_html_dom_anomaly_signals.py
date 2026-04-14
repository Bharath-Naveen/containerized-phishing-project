"""Tests for HTML/DOM anomaly extraction."""

from pathlib import Path

from bs4 import BeautifulSoup

from src.app_v1.html_dom_anomaly_signals import extract_html_dom_anomaly_signals
from src.app_v1.html_structure_signals import extract_html_structure_signals


def test_dom_anomaly_branded_anchor_and_external_form(tmp_path: Path) -> None:
    html = """
    <html><head><title>Google Sign-in</title></head>
    <body>
      <p>Please verify your account.</p>
      <a href="https://evil-login.example/submit">Sign in with Google</a>
      <form action="https://collect.bad/phish">
        <input type="email" name="e" />
        <input type="password" name="p" />
        <button type="submit">Continue</button>
      </form>
    </body></html>
    """
    p = tmp_path / "p.html"
    p.write_text(html, encoding="utf-8")
    out = extract_html_dom_anomaly_signals(
        html_path=str(p),
        final_url="https://landing.phish/page",
        input_url="https://landing.phish/page",
    )
    s = out["html_dom_anomaly_summary"]
    assert s is not None
    assert s["anchor_text_to_domain_mismatch_count"] >= 1
    assert s["form_action_external_domain_count"] >= 1
    assert s["title_brand_domain_mismatch"] is True
    assert out["html_dom_anomaly_risk_score"] is not None
    assert float(out["html_dom_anomaly_risk_score"]) > 0.25


def test_dom_anomaly_interstitial_phrases(tmp_path: Path) -> None:
    html = """
    <html><head><title>Link gateway</title></head>
    <body>
      <p>You will be redirected in a few seconds. Continue to your destination.</p>
      <a href="https://elsewhere.example/x">Click to continue</a>
    </body></html>
    """
    p = tmp_path / "g.html"
    p.write_text(html, encoding="utf-8")
    out = extract_html_dom_anomaly_signals(
        html_path=str(p),
        final_url="https://gate.example/",
        input_url="https://gate.example/",
    )
    s = out["html_dom_anomaly_summary"]
    assert s is not None
    assert s["interstitial_or_preview_pattern"] is True
    assert s["continue_to_destination_phrase_present"] is True


def test_same_ecosystem_links_not_counted_as_suspicious_external(tmp_path: Path) -> None:
    html = """
    <html><head><title>Medium home</title></head>
    <body>
      <a href="https://help.medium.com">Help</a>
      <a href="https://policy.medium.com">Policy</a>
      <a href="https://blog.medium.com">Blog</a>
      <p>Read stories and articles.</p>
    </body></html>
    """
    p = tmp_path / "m.html"
    p.write_text(html, encoding="utf-8")
    out = extract_html_dom_anomaly_signals(
        html_path=str(p),
        final_url="https://medium.com/",
        input_url="https://medium.com/",
    )
    s = out["html_dom_anomaly_summary"]
    assert s is not None
    assert s["same_ecosystem_external_links_suppressed"] >= 2
    assert s["anchor_text_to_domain_mismatch_count"] == 0
    assert s["wrapper_page_pattern"] is False


def test_dom_anomaly_missing_html_graceful() -> None:
    out = extract_html_dom_anomaly_signals(html_path=None, input_url="https://x.com")
    assert out["html_dom_anomaly_summary"] is None
    assert out["html_dom_anomaly_error"] == "missing_html_path"


def test_content_feed_does_not_force_impersonation(tmp_path: Path) -> None:
    """Forum/feed-like surface: many weak CTAs + brand names in prose should stay low-risk."""
    paras = " ".join(["Google and Microsoft discussed security updates in this thread."] * 25)
    lis = []
    for i in range(42):
        lis.append(
            f'<li><a href="https://out{i % 6}.newswire.net/article/{i}">Continue</a> '
            f"— replies and discussion #{i}</li>"
        )
    html = f"""<html><head><title>Community discussion</title></head>
    <body><article><h1>Weekly thread</h1><p>{paras}</p>
    <p>Posted by users. Permalink and pagination below.</p><ul>{"".join(lis)}</ul>
    </article></body></html>"""
    p = tmp_path / "forum.html"
    p.write_text(html, encoding="utf-8")
    out = extract_html_dom_anomaly_signals(
        html_path=str(p),
        final_url="https://community.example/threads/weekly",
        input_url="https://community.example/threads/weekly",
    )
    s = out["html_dom_anomaly_summary"]
    assert s is not None
    assert s["page_family"] == "content_feed_forum_aggregator"
    assert s["strict_anchor_filter_active"] is True
    assert s["anchor_text_to_domain_mismatch_count"] == 0
    assert s["generic_cta_external_link_count"] >= 1
    assert out["html_dom_visual_assessment"] != "suspicious_impersonation"
    assert float(out["html_dom_anomaly_risk_score"]) < 0.22


def test_shared_soup_single_parse(tmp_path: Path) -> None:
    html = "<html><body><form><input type=password /></form></body></html>"
    p = tmp_path / "s.html"
    p.write_text(html, encoding="utf-8")
    soup = BeautifulSoup(p.read_text(encoding="utf-8"), "html.parser")
    hs = extract_html_structure_signals(html_path=str(p), final_url="https://a.com/", input_url="https://a.com/", soup=soup)
    dom = extract_html_dom_anomaly_signals(html_path=str(p), final_url="https://a.com/", input_url="https://a.com/", soup=soup)
    assert hs["html_structure_summary"] is not None
    assert dom["html_dom_anomaly_summary"] is not None
