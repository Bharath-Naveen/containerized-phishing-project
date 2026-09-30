"""Record golden outputs of the dashboard analysis (rebuild Phase 2 safety net).

The Phase 2 refactor moves code around; it must not change what the app says. This script freezes
the full analysis payload for a fixed set of cases:

  * ~100 URLs through the ML-only path (suites, hard legit, official brand, PhishStats, plus tricky
    hosting/brand shapes), and
  * 8 synthetic live-capture cases (legit login, cross-domain credential form, wrapper page, blocked
    capture, content article, free-hosted brand clone, security block page, official authwall).

The ML model is not part of what is being protected, so its outputs are recorded once and replayed
(``predict_layer1`` and ``compute_layer1_model_agreement`` are patched). That keeps the golden file
independent of which model happens to be on disk.

Re-record ONLY when a behavior change is intended, and say why in docs/rebuild/REBUILD_LOG.md:
    PHISH_OUTPUTS_DIR=<dir with a model bundle> python tests/golden/record_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

GOLDEN = HERE / "dashboard_golden.json"
HTML_DIR = HERE / "html"


def golden_urls() -> list:
    from phishguard.data.eval_sets import (
        load_hard_legit_rows,
        load_official_brand_rows,
        load_phishstats_rows,
        load_url_suites,
    )

    urls = [u for v in load_url_suites().values() for u in v]
    urls += [r["url"] for r in load_hard_legit_rows()]
    urls += [r["url"] for r in load_official_brand_rows()][::15]
    urls += [r["url"] for r in load_phishstats_rows()][::10]
    urls += [
        "http://192.168.0.1/login.php", "http://xn--pypal-4ve.com/signin", "https://login.microsoftonline.com.secure-verify.info/",
        "https://my-site-123.weebly.com/", "https://outlook-verify.framer.app/", "https://docs.github.com/en/get-started",
        "https://www.linkedin.com/login", "https://www.virginatlantic.com/", "https://www.coursera.org/",
        "https://paypal-login.vercel.app/", "https://netflix-update-payment-details.vercel.app/", "https://sites.google.com/view/abc",
        "https://example.com/", "https://bit.ly/3xyz", "https://drive.google.com/file/d/abc/view", "http://user@evil.com@good.com/",
        "https://a.b.c.d.e.example.org/login", "https://secure-account-update.com/verify", "https://support.apple.com/",
        "https://appleid.apple.com.idmsa-login.co/",
    ]
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


CAPTURE_CASES = {
    "legit_login_same_domain": {
        "url": "https://www.example-bank.com/login",
        "final_url": "https://www.example-bank.com/login",
        "title": "Sign in | Example Bank",
        "html": "<html><head><title>Sign in | Example Bank</title></head><body><h1>Example Bank</h1>"
                "<form action='/session' method='post'><input name='user'><input type='password' name='pw'>"
                "<button>Sign in</button></form><a href='/help'>Help</a><a href='/privacy'>Privacy</a>"
                "<p>Welcome back to Example Bank online banking. Manage accounts, pay bills and more.</p></body></html>",
    },
    "cross_domain_credential_form": {
        "url": "https://paypal-secure-check.com/signin",
        "final_url": "https://paypal-secure-check.com/signin",
        "title": "PayPal: Log in",
        "html": "<html><head><title>PayPal: Log in</title></head><body><h1>PayPal</h1>"
                "<form action='https://collector-xyz.ru/p.php' method='post'><input name='email'>"
                "<input type='password' name='password'><button>Log In</button></form></body></html>",
        "network": ["https://collector-xyz.ru/p.php"],
    },
    "wrapper_interstitial": {
        "url": "https://share-docs-viewer.netlify.app/",
        "final_url": "https://share-docs-viewer.netlify.app/",
        "title": "Document shared with you",
        "html": "<html><head><title>Document shared with you</title><meta http-equiv='refresh' content='3;url=https://evil.example/login'>"
                "</head><body><p>You have received a secure document. Click to view.</p><a href='https://evil.example/login'>View document</a></body></html>",
    },
    "blocked_capture": {
        "url": "https://unknown-shop-deals.xyz/",
        "final_url": "",
        "title": "",
        "html": None,
        "error": "net::ERR_CONNECTION_REFUSED",
        "capture_blocked": True,
        "capture_strategy": "failed",
    },
    "content_article": {
        "url": "https://news.example.org/2026/04/story",
        "final_url": "https://news.example.org/2026/04/story",
        "title": "Local council approves new park | Example News",
        "html": "<html><head><title>Local council approves new park | Example News</title></head><body><article><h1>Local council approves new park</h1>"
                + "<p>The city council voted on Tuesday to approve a new riverside park. Residents spoke in favor.</p>" * 20
                + "</article><a href='/sports'>Sports</a><a href='/weather'>Weather</a></body></html>",
    },
    "free_hosted_brand_clone": {
        "url": "https://amazon-account-verify.github.io/",
        "final_url": "https://amazon-account-verify.github.io/",
        "title": "Amazon Sign-In",
        "html": "<html><head><title>Amazon Sign-In</title></head><body><h1>amazon</h1><form action='https://formspree.io/f/abc' method='post'>"
                "<input name='email'><input type='password' name='password'><button>Continue</button></form></body></html>",
    },
    "security_block_page": {
        "url": "https://suspicious-login-portal.com/",
        "final_url": "https://suspicious-login-portal.com/",
        "title": "Attention Required! | Cloudflare",
        "html": "<html><head><title>Attention Required! | Cloudflare</title></head><body><h1>Sorry, you have been blocked</h1>"
                "<p>This website is using a security service to protect itself from online attacks. Cloudflare Ray ID: 123</p></body></html>",
    },
    "official_authwall": {
        "url": "https://www.linkedin.com/in/someone",
        "final_url": "https://www.linkedin.com/authwall?trk=gf&sessionRedirect=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fsomeone",
        "title": "Sign Up | LinkedIn",
        "html": "<html><head><title>Sign Up | LinkedIn</title></head><body><h1>Join LinkedIn</h1><form action='/signup/cold-join' method='post'>"
                "<input name='email'><input type='password' name='password'><button>Agree &amp; Join</button></form></body></html>",
        "redirects": ["https://www.linkedin.com/in/someone", "https://www.linkedin.com/authwall"],
    },
}


def make_capture(case: dict):
    from phishguard.app.schemas import CaptureResult

    html_path = ""
    if case.get("html") is not None:
        HTML_DIR.mkdir(parents=True, exist_ok=True)
        name = case["url"].split("//", 1)[1].replace("/", "_")[:60] + ".html"
        p = HTML_DIR / name
        p.write_text(case["html"], encoding="utf-8")
        html_path = str(p.relative_to(REPO))
    visible = ""
    if case.get("html"):
        from bs4 import BeautifulSoup

        visible = BeautifulSoup(case["html"], "html.parser").get_text(" ", strip=True)
    redirects = case.get("redirects", [case["url"]] + ([case["final_url"]] if case.get("final_url") and case["final_url"] != case["url"] else []))
    return CaptureResult(
        original_url=case["url"], final_url=case.get("final_url", ""), title=case.get("title", ""),
        screenshot_path="", fullpage_screenshot_path="", html_path=html_path, visible_text=visible,
        initial_url=case["url"], redirect_chain=redirects, redirect_count=max(0, len(redirects) - 1),
        settled_successfully=case.get("error") is None, error=case.get("error"),
        capture_blocked=bool(case.get("capture_blocked", False)),
        capture_strategy=case.get("capture_strategy", "playwright_headless"),
        network_request_urls=list(case.get("network", [])),
        uses_https=case["url"].startswith("https://"),
    )


def normalize(payload: dict) -> dict:
    """Drop fields that legitimately differ between runs (time, absolute paths)."""
    drop_keys = {"timestamp_utc", "model_path", "path", "html_path", "analysis_json_path", "probability_calibration", "model_bundle"}

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items() if k not in drop_keys}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, float):
            return round(x, 9)
        return x

    return json.loads(json.dumps(walk(payload), default=str, sort_keys=True))


def run_case(dash_module, url: str, ml: dict, agreement: dict, capture=None) -> dict:
    from unittest.mock import patch

    def fake_predict(u, **kw):
        return json.loads(json.dumps(ml))

    def fake_agree(u, primary, **kw):
        return json.loads(json.dumps(agreement))

    with patch.object(dash_module, "predict_layer1", side_effect=fake_predict), \
         patch.object(dash_module, "compute_layer1_model_agreement", side_effect=fake_agree):
        if capture is None:
            payload, gaps = dash_module.build_dashboard_analysis(url, reinforcement=False)
        else:
            with patch.object(dash_module, "capture_url", return_value=capture):
                payload, gaps = dash_module.build_dashboard_analysis(url, reinforcement=True)
    return normalize({"payload": payload, "evidence_gaps": gaps})


def main() -> None:
    import phishguard.app.dashboard as dash
    from phishguard.app.ml_layer1 import compute_layer1_model_agreement, predict_layer1

    cases = []
    for url in golden_urls():
        ml = predict_layer1(url)
        agreement = compute_layer1_model_agreement(url, ml)
        ml = normalize(ml)
        agreement = normalize(agreement)
        cases.append({"kind": "ml_only", "url": url, "ml": ml, "agreement": agreement,
                      "expected": run_case(dash, url, ml, agreement)})
    for name, case in CAPTURE_CASES.items():
        ml = normalize(predict_layer1(case["url"]))
        agreement = normalize(compute_layer1_model_agreement(case["url"], ml))
        cases.append({"kind": "capture", "name": name, "url": case["url"], "ml": ml, "agreement": agreement,
                      "expected": run_case(dash, case["url"], ml, agreement, make_capture(case))})
    GOLDEN.write_text(json.dumps({"n_cases": len(cases), "cases": cases}, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {GOLDEN.relative_to(REPO)} with {len(cases)} cases")


if __name__ == "__main__":
    main()
