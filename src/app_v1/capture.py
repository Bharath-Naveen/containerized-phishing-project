"""Capture webpage evidence using Playwright with a fixed mobile viewport."""

from __future__ import annotations

import logging
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Literal, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from .config import PipelineConfig
from .schemas import CaptureInteractionMetadata, CaptureResult

logger = logging.getLogger(__name__)

CaptureNamespace = Literal["suspicious", "legit"]
_ALLOWED_CAPTURE_NAMESPACES = frozenset({"suspicious", "legit"})

DUMMY_EMAIL = "test.user@example.com"
DUMMY_PASSWORD = "NotARealPassword123!"

_CHROMIUM_STEALTH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
]

_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
"""

# Pool of mobile Safari/Chrome user agents for stealth / HTTP fallback.
_STEALTH_USER_AGENTS: List[str] = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; SM-S901B) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) CriOS/119.0.6045.109 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36",
]


class BlockedNavigationError(Exception):
    """Raised when page.goto fails with target blocking / timeout — caller may retry another strategy."""

    def __init__(self, reason_code: str, evidence: str):
        super().__init__(evidence)
        self.reason_code = reason_code
        self.evidence = evidence


# Locator strategies: first match wins; all failures are non-fatal.
_USER_INPUT_SELECTORS: List[str] = [
    'input[type="email"]:visible',
    'input[autocomplete="username"]:visible',
    'input[autocomplete="email"]:visible',
    'input[type="text"][name*="email" i]:visible',
    'input[type="text"][name*="user" i]:visible',
    'input[type="text"][name*="login" i]:visible',
    'input#email:visible',
    'input[name="username"]:visible',
    'input[name="email"]:visible',
    'input[type="text"]:visible',
]

_SUBMIT_SELECTORS: List[str] = [
    'button[type="submit"]:visible',
    'input[type="submit"]:visible',
]

_SUBMIT_BUTTON_NAME_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"sign\s*in", re.I),
    re.compile(r"log\s*in", re.I),
    re.compile(r"\blogin\b", re.I),
    re.compile(r"\bsubmit\b", re.I),
    re.compile(r"\bcontinue\b", re.I),
    re.compile(r"\bnext\b", re.I),
    re.compile(r"\bverify\b", re.I),
]

_CLICK_PROBE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\blogin\b", re.I),
    re.compile(r"sign\s*in", re.I),
    re.compile(r"\bcontinue\b", re.I),
    re.compile(r"\bverify\b", re.I),
    re.compile(r"\bnext\b", re.I),
]


# Labels for first_failed_capture_step and NotImplementedError messages (stable, human-readable).
STEP_SYNC_PLAYWRIGHT = "sync_playwright context"
STEP_BROWSER_LAUNCH = "browser launch (chromium.launch)"
STEP_BROWSER_CONTEXT = "browser context creation (new_context)"
STEP_PAGE_CREATE = "page creation (new_page)"
STEP_PAGE_GOTO = "navigation (page.goto)"
STEP_POST_LOAD_SLEEP = "post-load stabilization (time.sleep)"
STEP_PAGE_URL_FINAL = "read final URL (page.url)"
STEP_TITLE = "title extraction (page.title)"
STEP_PAGE_CONTENT = "html extraction (page.content)"
STEP_VISIBLE_TEXT = "visible text extraction (page.evaluate)"
STEP_HTML_WRITE = "html write to disk"
STEP_VIEWPORT_SCREENSHOT = "viewport screenshot (page.screenshot)"
STEP_FULLPAGE_SCREENSHOT = "full-page screenshot (page.screenshot)"
STEP_OPTIONAL_INTERACTION = "optional login interaction"
STEP_CONTEXT_CLOSE = "context close"
STEP_BROWSER_CLOSE = "browser close"
STEP_INTERACTION_LOCATE = "interaction: locate login fields (locator)"
STEP_INTERACTION_FILL = "interaction: fill credentials (locator.fill)"
STEP_INTERACTION_NAV = "interaction: submit navigation (click)"
STEP_INTERACTION_URL_AFTER = "interaction: page.url after submit"
STEP_INTERACTION_SCREENSHOT = "interaction: post-submit screenshot (page.screenshot)"


def _format_capture_exception(
    exc: BaseException,
    *,
    substep: Optional[str] = None,
) -> str:
    """Normalize exceptions for CaptureResult; never surface bare NotImplementedError."""
    name = type(exc).__name__
    msg = str(exc).strip() or "(no message)"
    if isinstance(exc, NotImplementedError):
        where = substep or "unknown Playwright sub-step"
        return (
            f"browser_feature_unsupported during {where}: sync Playwright raised NotImplementedError "
            f"({msg}). This usually indicates a Playwright/Chromium/greenlet or platform sync mismatch "
            "(not target-side blocking)."
        )
    return f"{name}: {msg}"


def _note_first_failure(first_fail: list[Optional[str]], step_label: str) -> None:
    if first_fail[0] is None:
        first_fail[0] = step_label


def _append_artifact_error(
    artifact_errors: list[str],
    first_fail: list[Optional[str]],
    step_label: str,
    exc: BaseException,
    *,
    error_key: str,
) -> None:
    _note_first_failure(first_fail, step_label)
    artifact_errors.append(
        f"{error_key}: {_format_capture_exception(exc, substep=step_label)}"
    )


def _is_blocked_navigation(exc: BaseException) -> bool:
    """Classify navigation failures commonly caused by anti-bot / headless detection or client blocks."""
    if isinstance(exc, PlaywrightTimeoutError):
        return True
    msg = f"{type(exc).__name__}: {exc}"
    upper = msg.upper()
    if "ERR_CONNECTION_ABORTED" in upper:
        return True
    if "ERR_BLOCKED_BY_CLIENT" in upper:
        return True
    if "ERR_TIMED_OUT" in upper:
        return True
    if "ERR_CONNECTION_RESET" in upper and "NET::" in upper:
        return True
    return False


def _playwright_block_reason(exc: BaseException) -> tuple[str, str]:
    msg = str(exc).strip() or "(no message)"
    upper = msg.upper()
    if "ERR_CONNECTION_ABORTED" in upper:
        return "playwright_navigation_aborted", f"Playwright navigation aborted by target: {msg}"
    if "ERR_BLOCKED_BY_CLIENT" in upper:
        return "playwright_navigation_aborted", f"Playwright navigation blocked by client/target policy: {msg}"
    if isinstance(exc, PlaywrightTimeoutError) or "ERR_TIMED_OUT" in upper:
        return "timeout", f"Playwright navigation timed out: {msg}"
    return "playwright_navigation_aborted", f"Playwright navigation failed in blocked pattern: {msg}"


def _http_block_reason(exc: BaseException) -> tuple[str, str]:
    msg = str(exc).strip() or "(no message)"
    if isinstance(exc, ssl.SSLError):
        return "http_tls_access_denied", f"Direct HTTP/TLS retrieval failed: {msg}"
    if isinstance(exc, urllib.error.HTTPError):
        code = int(getattr(exc, "code", 0) or 0)
        if code in {401, 403, 451}:
            return "http_blocked", f"Direct HTTP retrieval denied with HTTP {code}"
        return "http_blocked", f"Direct HTTP retrieval failed with HTTP {code}: {msg}"
    if isinstance(exc, urllib.error.URLError):
        reason_text = str(getattr(exc, "reason", msg))
        up = reason_text.upper()
        if "TLS" in up or "SSL" in up or "CERT" in up:
            return "http_tls_access_denied", f"Direct HTTP/TLS retrieval failed: {reason_text}"
        if "TIMED OUT" in up or "TIMEOUT" in up:
            return "timeout", f"Direct HTTP retrieval timed out: {reason_text}"
        return "http_blocked", f"Direct HTTP retrieval blocked/failed: {reason_text}"
    if isinstance(exc, TimeoutError):
        return "timeout", f"Direct HTTP retrieval timed out: {msg}"
    return "http_blocked", f"Direct HTTP retrieval failed: {msg}"


def _random_user_agent(fallback: str) -> str:
    try:
        return random.choice(_STEALTH_USER_AGENTS)
    except Exception:
        return fallback


@contextmanager
def _capture_step(
    logger_: logging.Logger,
    step: str,
    last_step: Optional[list[Optional[str]]] = None,
) -> Iterator[None]:
    """Log before/after a capture sub-step; optionally record active step for NotImplementedError."""
    if last_step is not None:
        last_step[0] = step
    logger_.info("capture: before %s", step)
    try:
        yield
    except Exception as exc:
        logger_.exception(
            "capture: step %s failed (%s): %s",
            step,
            type(exc).__name__,
            exc,
        )
        raise
    else:
        logger_.info("capture: after %s", step)


def _safe_locator_count(loc: Locator) -> int:
    """Playwright sync + greenlets can raise NotImplementedError from .count() on some setups."""
    try:
        return int(loc.count())
    except NotImplementedError as exc:
        logger.warning(
            "capture: locator.count() NotImplementedError (treating as 0): %s",
            exc,
        )
        return 0
    except PlaywrightError:
        return 0


def _capture_artifact_dir(output_root: Path, namespace: CaptureNamespace) -> Path:
    """Per-namespace subdirectory so suspicious and legit artifacts never collide."""
    if namespace not in _ALLOWED_CAPTURE_NAMESPACES:
        raise ValueError(
            f"capture namespace must be one of {sorted(_ALLOWED_CAPTURE_NAMESPACES)}, got {namespace!r}"
        )
    out = output_root / namespace
    out.mkdir(parents=True, exist_ok=True)
    return out


def _slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or "unknown-host"
    path = parsed.path.strip("/") or "root"
    raw = f"{host}-{path}"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-")
    return slug[:120] or "capture"


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _language_from_html_or_text(html: str, visible_text: str) -> tuple[Optional[str], Optional[str]]:
    lang_attr = None
    if html:
        m = re.search(r"<html[^>]*\blang=[\"']?([a-zA-Z-]+)", html, flags=re.I)
        if m:
            lang_attr = (m.group(1) or "").strip().lower()
    if lang_attr:
        return lang_attr.split("-")[0], "html_lang"
    txt = (visible_text or "").lower()
    if not txt:
        return None, None
    if any(w in txt for w in (" el ", " la ", " de ", " y ", " para ")):
        return "es", "text_guess"
    if any(w in txt for w in (" le ", " la ", " de ", " et ", " pour ")):
        return "fr", "text_guess"
    if any(w in txt for w in (" der ", " die ", " und ", " für ")):
        return "de", "text_guess"
    return "en", "text_guess"


def _locale_from_language(language: Optional[str]) -> tuple[str, str]:
    lang = (language or "").strip().lower()
    mapping = {
        "en": ("en-US", "en-US,en;q=0.9"),
        "es": ("es-ES", "es-ES,es;q=0.9,en;q=0.5"),
        "fr": ("fr-FR", "fr-FR,fr;q=0.9,en;q=0.5"),
        "de": ("de-DE", "de-DE,de;q=0.9,en;q=0.5"),
        "pt": ("pt-BR", "pt-BR,pt;q=0.9,en;q=0.5"),
    }
    return mapping.get(lang, ("en-US", "en-US,en;q=0.9"))


def _extract_visible_text(page: Any, max_chars: int) -> str:
    text = page.evaluate(
        """() => {
            if (!document || !document.body) return "";
            return (document.body.innerText || "").trim();
        }"""
    )
    return (text or "")[:max_chars]


def _title_and_text_from_html(html: str, max_text_chars: int) -> tuple[str, str]:
    try:
        soup = BeautifulSoup(html, "html.parser")
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return title, (text or "")[:max_text_chars]
    except Exception:
        return "", ""


def _first_matching_locator(page: Any, selectors: List[str]) -> Optional[Locator]:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if _safe_locator_count(loc) > 0:
                return loc.first
        except PlaywrightError:
            continue
    return None


def _find_user_input(page: Any) -> Optional[Locator]:
    return _first_matching_locator(page, _USER_INPUT_SELECTORS)


def _find_password_input(page: Any) -> Optional[Locator]:
    try:
        loc = page.locator('input[type="password"]:visible')
        if _safe_locator_count(loc) > 0:
            return loc.first
    except PlaywrightError:
        pass
    return None


def _find_submit_control(page: Any) -> Optional[Locator]:
    submit = _first_matching_locator(page, _SUBMIT_SELECTORS)
    if submit is not None:
        return submit
    for pattern in _SUBMIT_BUTTON_NAME_PATTERNS:
        try:
            btn = page.get_by_role("button", name=pattern)
            if _safe_locator_count(btn) > 0:
                return btn.first
        except PlaywrightError:
            continue
    return None


def _stabilize_after_load_ms(ms: int) -> None:
    """Avoid page.wait_for_timeout: it can raise NotImplementedError in sync Playwright on some OS/driver combos."""
    if ms <= 0:
        return
    time.sleep(ms / 1000.0)


def _run_optional_login_interaction(
    page: Any,
    cfg: PipelineConfig,
    slug: str,
    output_dir: Path,
    first_fail: list[Optional[str]],
    last_step: Optional[list[Optional[str]]],
) -> CaptureInteractionMetadata:
    """Best-effort dummy login interaction; never raises."""
    meta = CaptureInteractionMetadata()
    try:
        with _capture_step(logger, STEP_INTERACTION_LOCATE, last_step):
            user_loc = _find_user_input(page)
            pw_loc = _find_password_input(page)
        meta.user_field_found = user_loc is not None
        meta.password_field_found = pw_loc is not None

        if user_loc is None or pw_loc is None:
            return meta

        submit_loc = _find_submit_control(page)
        if submit_loc is None:
            return meta

        fill_timeout = min(5000, max(2000, cfg.navigation_timeout_ms // 6))

        with _capture_step(logger, STEP_INTERACTION_FILL, last_step):
            user_loc.fill(DUMMY_EMAIL, timeout=fill_timeout)
            pw_loc.fill(DUMMY_PASSWORD, timeout=fill_timeout)

        with _capture_step(logger, STEP_INTERACTION_NAV, last_step):
            meta.url_before_submit = page.url
            submit_loc.click(timeout=fill_timeout)
        meta.submit_found_and_clicked = True
        meta.attempted_submit = True

        stabilize = max(0, cfg.post_submit_stabilize_ms)
        _stabilize_after_load_ms(stabilize)

        with _capture_step(logger, STEP_INTERACTION_URL_AFTER, last_step):
            meta.url_after_submit = page.url
        meta.navigation_occurred = bool(
            meta.url_before_submit
            and meta.url_after_submit
            and meta.url_before_submit != meta.url_after_submit
        )

        post_path = output_dir / f"{slug}-after-submit.png"
        with _capture_step(logger, STEP_INTERACTION_SCREENSHOT, last_step):
            page.screenshot(path=str(post_path), full_page=False)
        meta.post_submit_viewport_screenshot_path = str(post_path)
    except Exception as exc:  # noqa: BLE001 — interaction must never fail the pipeline.
        sub = last_step[0] if last_step else STEP_OPTIONAL_INTERACTION
        _note_first_failure(first_fail, sub)
        meta.interaction_error = _format_capture_exception(exc, substep=sub)
    return meta


def _iter_click_probe_candidates(page: Any) -> List[tuple[Locator, str]]:
    """Collect visible controls whose text matches safe probe patterns (order stable)."""
    out: List[tuple[Locator, str]] = []
    selectors = [
        'button:visible',
        'a:visible',
        'input[type="button"]:visible',
        'input[type="submit"]:visible',
        '[role="button"]:visible',
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            n = min(_safe_locator_count(loc), 20)
            for i in range(n):
                candidate = loc.nth(i)
                txt = ""
                try:
                    txt = (candidate.inner_text(timeout=300) or "").strip()
                except Exception:
                    try:
                        txt = (candidate.get_attribute("value") or "").strip()
                    except Exception:
                        txt = ""
                if not txt:
                    continue
                if any(p.search(txt) for p in _CLICK_PROBE_PATTERNS):
                    out.append((candidate, txt[:120]))
        except PlaywrightError:
            continue
    return out


def _find_click_probe_control(page: Any) -> tuple[Optional[Locator], Optional[str]]:
    """Return first visible safe click target and its text (best effort)."""
    found = _iter_click_probe_candidates(page)
    if not found:
        return None, None
    loc, txt = found[0]
    return loc, txt


def _run_optional_click_probe(
    page: Any,
    cfg: PipelineConfig,
    first_fail: list[Optional[str]],
    last_step: Optional[list[Optional[str]]],
) -> CaptureInteractionMetadata:
    """Safe click probe: one click only, no typing."""
    meta = CaptureInteractionMetadata()
    meta.click_probe_enabled = bool(cfg.enable_click_probe)
    if not cfg.enable_click_probe:
        meta.click_probe_skip_reason = "disabled_by_config"
        meta.click_probe_attempted = False
        meta.click_probe_candidate_count = 0
        meta.click_probe_candidate_texts_sample = []
        return meta
    try:
        candidates = _iter_click_probe_candidates(page)
    except Exception:  # noqa: BLE001 — probe must never fail the pipeline
        meta.click_probe_skip_reason = "html_or_page_unavailable"
        meta.click_probe_attempted = False
        meta.click_probe_candidate_count = 0
        meta.click_probe_candidate_texts_sample = []
        return meta

    meta.click_probe_candidate_count = len(candidates)
    meta.click_probe_candidate_texts_sample = [t for _, t in candidates[:5]]
    meta.click_probe_attempted = True
    if not candidates:
        meta.click_probe_skip_reason = "no_visible_candidate"
        return meta

    target, txt = candidates[0]
    meta.click_probe_text = txt
    meta.click_probe_before_url = page.url
    try:
        with _capture_step(logger, STEP_INTERACTION_NAV, last_step):
            target.click(timeout=min(5000, max(2000, cfg.navigation_timeout_ms // 6)))
        _stabilize_after_load_ms(max(0, cfg.post_submit_stabilize_ms))
        with _capture_step(logger, STEP_INTERACTION_URL_AFTER, last_step):
            meta.click_probe_after_url = page.url
        meta.click_probe_domain_changed = bool(
            _host(meta.click_probe_before_url or "") and _host(meta.click_probe_after_url or "")
            and _host(meta.click_probe_before_url or "") != _host(meta.click_probe_after_url or "")
        )
        meta.click_probe_skip_reason = None
    except PlaywrightTimeoutError as exc:  # noqa: BLE001
        sub = last_step[0] if last_step else STEP_OPTIONAL_INTERACTION
        _note_first_failure(first_fail, sub)
        meta.click_probe_skip_reason = "candidate_not_clickable"
        meta.click_probe_error = _format_capture_exception(exc, substep=sub)
    except Exception as exc:  # noqa: BLE001
        sub = last_step[0] if last_step else STEP_OPTIONAL_INTERACTION
        _note_first_failure(first_fail, sub)
        meta.click_probe_skip_reason = "probe_error"
        meta.click_probe_error = _format_capture_exception(exc, substep=sub)
    return meta


def _http_fetch_html(url: str, timeout_s: float) -> tuple[str, str]:
    """Fetch raw HTML via HTTP(S). Returns (final_url, html). Raises on failure."""
    ua = _random_user_agent(_STEALTH_USER_AGENTS[0])
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        final_url = resp.geturl()
        raw = resp.read()
    html = raw.decode("utf-8", errors="replace")
    return final_url, html


def _playwright_full_capture(
    url: str,
    cfg: PipelineConfig,
    output_dir: Path,
    slug: str,
    viewport_png: Path,
    fullpage_png: Path,
    html_path: Path,
    *,
    headless: bool,
    user_agent: str,
    stealth: bool,
    capture_strategy_label: str,
    preferred_language: Optional[str] = None,
) -> CaptureResult:
    """Run Playwright: goto + artifacts. Raises BlockedNavigationError on blocked-class goto timeouts."""
    logger.info(
        "capture: playwright attempt strategy=%s headless=%s stealth=%s",
        capture_strategy_label,
        headless,
        stealth,
    )
    last_step: list[Optional[str]] = ["(unknown)"]
    first_fail: list[Optional[str]] = [None]

    def _fail_result_from_nav(nav_exc: BaseException) -> CaptureResult:
        _note_first_failure(first_fail, STEP_PAGE_GOTO)
        return CaptureResult(
            original_url=url,
            final_url=url,
            title="",
            screenshot_path="",
            fullpage_screenshot_path="",
            html_path="",
            visible_text="",
            interaction=CaptureInteractionMetadata(),
            error=(
                f"capture_failed: {_format_capture_exception(nav_exc, substep=STEP_PAGE_GOTO)}"
            ),
            capture_blocked=False,
            capture_strategy="failed",
            capture_block_reason=None,
            capture_block_evidence=None,
            first_failed_capture_step=STEP_PAGE_GOTO,
        )

    try:
        with _capture_step(logger, STEP_SYNC_PLAYWRIGHT, last_step):
            with sync_playwright() as playwright:
                browser = None
                context = None
                try:
                    launch_kwargs: dict[str, Any] = {"headless": headless}
                    if stealth:
                        launch_kwargs["args"] = list(_CHROMIUM_STEALTH_ARGS)

                    with _capture_step(logger, STEP_BROWSER_LAUNCH, last_step):
                        browser = playwright.chromium.launch(**launch_kwargs)

                    with _capture_step(logger, STEP_BROWSER_CONTEXT, last_step):
                        locale, accept_language = _locale_from_language(
                            preferred_language
                        )
                        context = browser.new_context(
                            viewport={
                                "width": cfg.fixed_viewport_width,
                                "height": cfg.fixed_viewport_height,
                            },
                            user_agent=user_agent,
                            locale=locale,
                            timezone_id="America/New_York",
                            is_mobile=True,
                            device_scale_factor=2,
                            has_touch=True,
                            color_scheme="light",
                            extra_http_headers={"Accept-Language": accept_language},
                        )

                    with _capture_step(logger, STEP_PAGE_CREATE, last_step):
                        page = context.new_page()
                        nav_chain: list[str] = []
                        seen_nav: set[str] = set()
                        network_requests: List[str] = []
                        seen_requests: set[str] = set()

                        def _track_nav(frame: Any) -> None:
                            try:
                                if frame != page.main_frame:
                                    return
                                u = str(frame.url or "")
                                if u and u not in seen_nav:
                                    seen_nav.add(u)
                                    nav_chain.append(u)
                            except Exception:
                                return

                        page.on("framenavigated", _track_nav)
                        def _track_request(req: Any) -> None:
                            try:
                                u = str(req.url or "")
                            except Exception:
                                u = ""
                            if not u:
                                return
                            if u in seen_requests:
                                return
                            seen_requests.add(u)
                            if len(network_requests) < 400:
                                network_requests.append(u)
                        page.on("request", _track_request)
                        if stealth:
                            page.add_init_script(_STEALTH_INIT_SCRIPT)

                    with _capture_step(logger, STEP_PAGE_GOTO, last_step):
                        try:
                            page.goto(
                                url,
                                wait_until=cfg.wait_until,
                                timeout=cfg.navigation_timeout_ms,
                            )
                        except Exception as nav_exc:
                            if _is_blocked_navigation(nav_exc):
                                code, evidence = _playwright_block_reason(nav_exc)
                                logger.warning(
                                    "capture: navigation classified as capture_blocked_by_target code=%s (%s)",
                                    code,
                                    evidence,
                                )
                                raise BlockedNavigationError(code, evidence) from nav_exc
                            return _fail_result_from_nav(nav_exc)

                    with _capture_step(logger, STEP_POST_LOAD_SLEEP, last_step):
                        settle_start = time.monotonic()
                        settled_successfully = True
                        try:
                            page.wait_for_load_state(
                                "networkidle",
                                timeout=min(cfg.navigation_timeout_ms, 8000),
                            )
                        except Exception:
                            settled_successfully = False
                        _stabilize_after_load_ms(cfg.post_load_stabilize_ms)
                        settle_time_ms = int((time.monotonic() - settle_start) * 1000)

                    artifact_errors: list[str] = []

                    with _capture_step(logger, STEP_PAGE_URL_FINAL, last_step):
                        try:
                            final_url = page.url
                        except Exception as exc:  # noqa: BLE001
                            final_url = url
                            _append_artifact_error(
                                artifact_errors,
                                first_fail,
                                STEP_PAGE_URL_FINAL,
                                exc,
                                error_key="final_url_failed",
                            )
                    if final_url and (not nav_chain or nav_chain[-1] != final_url):
                        nav_chain.append(final_url)
                    if not nav_chain:
                        nav_chain = [url, final_url] if final_url and final_url != url else [url]
                    redirect_count = max(0, len(nav_chain) - 1)
                    cross_domain_redirect_count = 0
                    for i in range(1, len(nav_chain)):
                        if _host(nav_chain[i]) and _host(nav_chain[i - 1]) and _host(nav_chain[i]) != _host(nav_chain[i - 1]):
                            cross_domain_redirect_count += 1

                    title = ""
                    with _capture_step(logger, STEP_TITLE, last_step):
                        try:
                            title = page.title()
                        except Exception as exc:  # noqa: BLE001
                            _append_artifact_error(
                                artifact_errors,
                                first_fail,
                                STEP_TITLE,
                                exc,
                                error_key="title_extraction_failed",
                            )

                    html = ""
                    with _capture_step(logger, STEP_PAGE_CONTENT, last_step):
                        try:
                            html = page.content()
                        except Exception as exc:  # noqa: BLE001
                            _append_artifact_error(
                                artifact_errors,
                                first_fail,
                                STEP_PAGE_CONTENT,
                                exc,
                                error_key="html_capture_failed",
                            )

                    visible_text = ""
                    with _capture_step(logger, STEP_VISIBLE_TEXT, last_step):
                        try:
                            visible_text = _extract_visible_text(
                                page, cfg.visible_text_max_chars
                            )
                        except Exception as exc:  # noqa: BLE001
                            _append_artifact_error(
                                artifact_errors,
                                first_fail,
                                STEP_VISIBLE_TEXT,
                                exc,
                                error_key="text_extraction_failed",
                            )

                    written_html_path = ""
                    if html:
                        with _capture_step(logger, STEP_HTML_WRITE, last_step):
                            try:
                                html_path.write_text(
                                    html, encoding="utf-8", errors="ignore"
                                )
                                if html_path.exists():
                                    written_html_path = str(html_path)
                                else:
                                    _note_first_failure(first_fail, STEP_HTML_WRITE)
                                    artifact_errors.append(
                                        "html_write_failed: file was not found after write"
                                    )
                            except Exception as exc:  # noqa: BLE001
                                _append_artifact_error(
                                    artifact_errors,
                                    first_fail,
                                    STEP_HTML_WRITE,
                                    exc,
                                    error_key="html_write_failed",
                                )
                    else:
                        artifact_errors.append("html_write_skipped: no html captured")

                    written_viewport_path = ""
                    with _capture_step(logger, STEP_VIEWPORT_SCREENSHOT, last_step):
                        try:
                            page.screenshot(path=str(viewport_png), full_page=False)
                            if viewport_png.exists():
                                written_viewport_path = str(viewport_png)
                            else:
                                _note_first_failure(
                                    first_fail, STEP_VIEWPORT_SCREENSHOT
                                )
                                artifact_errors.append(
                                    "viewport_screenshot_failed: file was not found after write"
                                )
                        except Exception as exc:  # noqa: BLE001
                            _append_artifact_error(
                                artifact_errors,
                                first_fail,
                                STEP_VIEWPORT_SCREENSHOT,
                                exc,
                                error_key="viewport_screenshot_failed",
                            )

                    written_fullpage_path = ""
                    with _capture_step(logger, STEP_FULLPAGE_SCREENSHOT, last_step):
                        try:
                            page.screenshot(path=str(fullpage_png), full_page=True)
                            if fullpage_png.exists():
                                written_fullpage_path = str(fullpage_png)
                            else:
                                _note_first_failure(
                                    first_fail, STEP_FULLPAGE_SCREENSHOT
                                )
                                artifact_errors.append(
                                    "fullpage_screenshot_failed: file was not found after write"
                                )
                        except Exception as exc:  # noqa: BLE001
                            _append_artifact_error(
                                artifact_errors,
                                first_fail,
                                STEP_FULLPAGE_SCREENSHOT,
                                exc,
                                error_key="fullpage_screenshot_failed",
                            )

                    with _capture_step(logger, STEP_OPTIONAL_INTERACTION, last_step):
                        interaction = _run_optional_login_interaction(
                            page, cfg, slug, output_dir, first_fail, last_step
                        )
                        probe = _run_optional_click_probe(page, cfg, first_fail, last_step)
                        for k, v in probe.as_json().items():
                            if k.startswith("click_probe"):
                                setattr(interaction, k, v)
                    detected_language, language_source = _language_from_html_or_text(
                        html, visible_text
                    )

                    capture_blocked = capture_strategy_label != "playwright_headless"
                    return CaptureResult(
                        original_url=url,
                        final_url=final_url,
                        title=title,
                        screenshot_path=written_viewport_path,
                        fullpage_screenshot_path=written_fullpage_path,
                        html_path=written_html_path,
                        visible_text=visible_text,
                        initial_url=url,
                        redirect_chain=nav_chain,
                        redirect_count=redirect_count,
                        cross_domain_redirect_count=cross_domain_redirect_count,
                        settle_time_ms=settle_time_ms,
                        settled_successfully=settled_successfully,
                        detected_language=detected_language,
                        language_source=language_source,
                        interaction=interaction,
                        error=" | ".join(artifact_errors)
                        if artifact_errors
                        else None,
                        capture_blocked=capture_blocked,
                        capture_strategy=capture_strategy_label,
                        capture_block_reason=None,
                        capture_block_evidence=None,
                        first_failed_capture_step=first_fail[0],
                            network_request_urls=network_requests,
                    )
                finally:
                    if context is not None:
                        with _capture_step(logger, STEP_CONTEXT_CLOSE, last_step):
                            context.close()
                    if browser is not None:
                        with _capture_step(logger, STEP_BROWSER_CLOSE, last_step):
                            browser.close()
    except NotImplementedError as nie:
        logger.exception(
            "capture: uncaught NotImplementedError at step=%s",
            last_step[0],
        )
        return CaptureResult(
            original_url=url,
            final_url=url,
            title="",
            screenshot_path="",
            fullpage_screenshot_path="",
            html_path="",
            visible_text="",
            interaction=CaptureInteractionMetadata(),
            error=f"capture_failed: {_format_capture_exception(nie, substep=last_step[0])}",
            capture_blocked=False,
            capture_strategy="failed",
            capture_block_reason=None,
            capture_block_evidence=None,
            first_failed_capture_step=last_step[0],
        )


def capture_url(
    url: str,
    config: Optional[PipelineConfig] = None,
    *,
    namespace: CaptureNamespace = "suspicious",
    preferred_language: Optional[str] = None,
) -> CaptureResult:
    """Capture viewport + full-page screenshots, HTML, optional login interaction.

    Fallback order: headless Playwright → stealth Playwright → HTTP HTML only → evasive error.
    """
    cfg = config or PipelineConfig.from_env()
    base_output = cfg.ensure_output_dir()
    output_dir = _capture_artifact_dir(base_output, namespace)
    slug = _slug_from_url(url)
    viewport_png = output_dir / f"{slug}.png"
    fullpage_png = output_dir / f"{slug}-fullpage.png"
    html_path = output_dir / f"{slug}.html"

    timeout_s = max(5.0, cfg.navigation_timeout_ms / 1000.0)

    logger.info("capture: entering pipeline for url=%s namespace=%s", url, namespace)
    playwright_reasons: list[str] = []
    playwright_evidence: list[str] = []

    # Strategy 0 — standard headless (no anti-detection extras).
    try:
        return _playwright_full_capture(
            url,
            cfg,
            output_dir,
            slug,
            viewport_png,
            fullpage_png,
            html_path,
            headless=True,
            user_agent=cfg.user_agent,
            stealth=False,
            capture_strategy_label="playwright_headless",
            preferred_language=preferred_language,
        )
    except BlockedNavigationError as first_block:
        logger.warning(
            "capture: headless blocked reason=%s evidence=%s; trying stealth",
            first_block.reason_code,
            first_block.evidence,
        )
        playwright_reasons.append(first_block.reason_code)
        playwright_evidence.append(first_block.evidence)
    except Exception as exc:  # noqa: BLE001
        return CaptureResult(
            original_url=url,
            final_url=url,
            title="",
            screenshot_path="",
            fullpage_screenshot_path="",
            html_path="",
            visible_text="",
            interaction=CaptureInteractionMetadata(),
            error=f"capture_failed: {_format_capture_exception(exc)}",
            capture_blocked=False,
            capture_strategy="failed",
            capture_block_reason=None,
            capture_block_evidence=None,
            first_failed_capture_step=None,
            network_request_urls=[],
        )

    # Strategy A — stealth (headed, randomized UA, automation flags reduced).
    stealth_ua = _random_user_agent(cfg.user_agent)
    try:
        return _playwright_full_capture(
            url,
            cfg,
            output_dir,
            slug,
            viewport_png,
            fullpage_png,
            html_path,
            headless=False,
            user_agent=stealth_ua,
            stealth=True,
            capture_strategy_label="playwright_stealth",
            preferred_language=preferred_language,
        )
    except BlockedNavigationError as stealth_block:
        logger.warning(
            "capture: stealth Playwright also blocked reason=%s evidence=%s; trying HTTP HTML fallback",
            stealth_block.reason_code,
            stealth_block.evidence,
        )
        playwright_reasons.append(stealth_block.reason_code)
        playwright_evidence.append(stealth_block.evidence)
    except Exception as exc:  # noqa: BLE001
        logger.exception("capture: stealth attempt failed with non-navigation error")
        # Continue to HTTP fallback if we still have nothing; else surface error
        pass

    # Strategy B — HTTP-only HTML (no browser).
    try:
        with _capture_step(logger, "http fallback (urllib) fetch"):
            final_url, html = _http_fetch_html(url, timeout_s=timeout_s)
        title, visible_text = _title_and_text_from_html(html, cfg.visible_text_max_chars)
        written_html_path = ""
        write_err: Optional[str] = None
        try:
            html_path.write_text(html, encoding="utf-8", errors="ignore")
            if html_path.exists():
                written_html_path = str(html_path)
        except Exception as wexc:  # noqa: BLE001
            logger.exception("capture: http fallback html write failed: %s", wexc)
            write_err = _format_capture_exception(wexc)

        return CaptureResult(
            original_url=url,
            final_url=final_url or url,
            title=title,
            screenshot_path="",
            fullpage_screenshot_path="",
            html_path=written_html_path,
            visible_text=visible_text,
            initial_url=url,
            redirect_chain=[url, final_url] if final_url and final_url != url else [url],
            redirect_count=1 if final_url and final_url != url else 0,
            cross_domain_redirect_count=(
                1
                if final_url
                and final_url != url
                and _host(final_url) != _host(url)
                else 0
            ),
            settle_time_ms=0,
            settled_successfully=False,
            detected_language=_language_from_html_or_text(html, visible_text)[0],
            language_source=_language_from_html_or_text(html, visible_text)[1],
            interaction=CaptureInteractionMetadata(),
            error=(
                f"capture_failed: html_write_failed: {write_err}"
                if write_err
                else None
            ),
            capture_blocked=True,
            capture_strategy="http_fallback",
            capture_block_reason=(
                " + ".join(dict.fromkeys(playwright_reasons))
                if playwright_reasons
                else "http_fallback_used"
            ),
            capture_block_evidence=(
                "; ".join(dict.fromkeys(playwright_evidence))
                if playwright_evidence
                else "Playwright capture was unavailable; direct HTTP retrieval succeeded."
            ),
            first_failed_capture_step=None,
            network_request_urls=[],
        )
    except Exception as httpe:  # noqa: BLE001
        logger.exception("capture: HTTP fallback failed: %s", httpe)
        http_reason, http_evidence = _http_block_reason(httpe)
        all_reasons = list(dict.fromkeys([*playwright_reasons, http_reason]))
        all_evidence = list(dict.fromkeys([*playwright_evidence, http_evidence]))
        combined_reason = " + ".join(all_reasons) if all_reasons else "http_blocked"
        combined_evidence = (
            "; ".join(all_evidence)
            if all_evidence
            else "Browser automation and direct HTTP retrieval were both blocked."
        )

        # Strategy C — evasive / fully blocked.
        return CaptureResult(
            original_url=url,
            final_url=url,
            title="",
            screenshot_path="",
            fullpage_screenshot_path="",
            html_path="",
            visible_text="",
            initial_url=url,
            redirect_chain=[url],
            redirect_count=0,
            cross_domain_redirect_count=0,
            settle_time_ms=0,
            settled_successfully=False,
            detected_language=None,
            language_source=None,
            interaction=CaptureInteractionMetadata(),
            error=(
                "capture_blocked_by_target: site likely blocking automated browsing"
            ),
            capture_blocked=True,
            capture_strategy="failed",
            capture_block_reason=combined_reason,
            capture_block_evidence=combined_evidence,
            first_failed_capture_step=None,
            network_request_urls=[],
        )
