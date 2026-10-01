"""Passive Playwright navigation for behavior signals (optional)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_CHALLENGE_SNIPPETS = (
    "cf-browser-verification",
    "checking your browser",
    "attention required",
    "cloudflare",
    "captcha",
    "enable javascript",
    "bot detection",
)


def run_behavior_probe(
    url: str,
    *,
    screenshot_dir: Path | None,
    navigation_timeout_ms: int = 20000,
) -> Dict[str, Any]:
    """Ephemeral context; no downloads; no form submission."""
    out: Dict[str, Any] = {
        "page_load_success": 0,
        "automation_blocked": 0,
        "screenshot_captured": 0,
        "final_url_differs_from_input": 0,
        "navigation_timeout": 0,
        "challenge_detected": 0,
        "behavior_error": "",
    }
    input_host = (urlsplit(url).hostname or "").lower()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        out["behavior_error"] = f"playwright_import:{e}"
        return out

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--no-sandbox"],
            )
            context = browser.new_context(
                accept_downloads=False,
                java_script_enabled=True,
                ignore_https_errors=True,
            )
            context.clear_cookies()
            page = context.new_page()
            page.set_default_navigation_timeout(navigation_timeout_ms)
            page.set_default_timeout(navigation_timeout_ms)
            try:
                resp = page.goto(url, wait_until="domcontentloaded")
                out["page_load_success"] = int(resp is not None and resp.ok)
            except Exception as e:
                msg = str(e).lower()
                out["behavior_error"] = type(e).__name__
                if "timeout" in msg:
                    out["navigation_timeout"] = 1
                if "blocked" in msg or "aborted" in msg:
                    out["automation_blocked"] = 1
                browser.close()
                return out

            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            time.sleep(0.3)

            final = page.url
            out["final_url_differs_from_input"] = int(final.rstrip("/") != url.rstrip("/"))
            fh = (urlsplit(final).hostname or "").lower()
            if input_host and fh and input_host != fh:
                out["final_url_differs_from_input"] = 1

            html_l = (page.content() or "").lower()
            if any(s in html_l for s in _CHALLENGE_SNIPPETS):
                out["challenge_detected"] = 1
                out["automation_blocked"] = 1

            if screenshot_dir is not None:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                path = screenshot_dir / "behavior_probe.png"
                try:
                    page.screenshot(path=str(path), full_page=False)
                    out["screenshot_captured"] = int(path.exists())
                except Exception as se:
                    logger.debug("screenshot failed: %s", se)
            context.close()
            browser.close()
    except Exception as e:
        out["behavior_error"] = f"{type(e).__name__}:{e}"
    return out


def merge_behavior(
    http_fetch_ok: int,
    dom_missing: int,
    url: str,
    *,
    use_playwright: bool,
    artifact_root: Path | None,
) -> Dict[str, Any]:
    if not use_playwright:
        return {
            "page_load_success": int(bool(http_fetch_ok)),
            "automation_blocked": 0,
            "screenshot_captured": 0,
            "final_url_differs_from_input": 0,
            "navigation_timeout": 0,
            "challenge_detected": 0,
            "behavior_skipped": 1,
            "behavior_error": "",
        }
    shot_dir = (artifact_root / "behavior_shots") if artifact_root else None
    b = run_behavior_probe(url, screenshot_dir=shot_dir)
    b["behavior_skipped"] = 0
    if http_fetch_ok and not b.get("page_load_success"):
        b["automation_blocked"] = max(b.get("automation_blocked", 0), 0)
    return b
