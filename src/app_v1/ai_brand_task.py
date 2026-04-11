"""Infer likely brand and task from screenshot + page text using OpenAI."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from openai import OpenAI

from .config import PipelineConfig
from .schemas import AIBrandTaskResult, URLIntelResult


def normalize_task_guess(
    task_guess: str,
    *,
    password_field_present: bool,
    final_url: str,
    visible_text: str,
) -> str:
    """Broaden task labels so sign-in *links* without an in-page password flow are not labeled login.

    Retail and portal homepages often surface "Sign in" in the chrome while the primary
    content is browsing; Playwright interaction only sees fields in the viewport.
    """
    raw = (task_guess or "unknown").strip().lower() or "unknown"
    if password_field_present:
        return raw

    login_like = {"login", "unknown"}
    # "informational" on a shallow URL with lots of chrome text is often a storefront home.
    if raw not in login_like and raw != "informational":
        return raw

    try:
        parsed = urlparse(final_url or "")
        path = (parsed.path or "").rstrip("/")
    except Exception:
        path = ""
    shallow_path = path == "" or path == "/"
    text_len = len((visible_text or "").strip())
    long_page_text = text_len > 1000

    if shallow_path or long_page_text:
        return "homepage"

    return raw


def _to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"screenshot not found: {image_path}")

    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix)
    if not mime_type:
        raise ValueError(f"unsupported screenshot type: {suffix}")

    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def _screenshot_is_usable(image_path: str) -> bool:
    if not image_path:
        return False
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return False
    suffix = path.suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".webp"}


def _extract_text(response_obj: Any) -> str:
    # SDK object may expose output_text directly; fallback to JSON string form.
    txt = getattr(response_obj, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    if hasattr(response_obj, "model_dump_json"):
        return response_obj.model_dump_json(indent=2)
    return str(response_obj)


def _parse_payload(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def infer_brand_and_task(
    screenshot_path: str,
    visible_text: str,
    title: str,
    final_url: str,
    config: Optional[PipelineConfig] = None,
    *,
    password_field_present: bool = False,
    url_intel: Optional[URLIntelResult] = None,
) -> AIBrandTaskResult:
    """Call Responses API and return strict, normalized JSON fields."""
    cfg = config or PipelineConfig.from_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return AIBrandTaskResult(error="missing OPENAI_API_KEY")

    prompt = (
        "You are a phishing triage classifier.\n"
        "Use the screenshot and page context to infer a likely brand and user task.\n\n"
        "Return ONLY strict JSON with this exact schema:\n"
        '{\n'
        '  "brand_guess": "string",\n'
        '  "task_guess": "string",\n'
        '  "reasons": ["string"],\n'
        '  "unknown": true\n'
        "}\n\n"
        "Rules:\n"
        "- If uncertain, set unknown=true and use 'unknown' for brand/task as needed.\n"
        "- task_guess should be one of: login, homepage, browse, checkout, password reset, "
        "account verification, document share, informational, unknown.\n"
        "- Use homepage or browse when the main experience is shopping or browsing, even if a "
        "sign-in link appears in the header; use login only when sign-in or credential entry "
        "is the primary focus of the page.\n"
        "- No markdown, no extra keys."
    )

    context_text = (
        f"Page title: {title}\n"
        f"Final URL: {final_url}\n"
        f"URL intel hints: brand={getattr(url_intel, 'brand_hint', 'unknown')}, "
        f"product={getattr(url_intel, 'product_hint', 'unknown')}, "
        f"action={getattr(url_intel, 'action_hint', 'unknown')}, "
        f"plausibility={getattr(url_intel, 'first_party_url_plausibility', 'unknown')}\n"
        f"Visible text excerpt:\n{(visible_text or '')[:cfg.ai_text_max_chars]}"
    )

    try:
        client = OpenAI(api_key=api_key)
        user_content = [{"type": "input_text", "text": context_text}]
        screenshot_used = _screenshot_is_usable(screenshot_path)
        if screenshot_used:
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": _to_data_url(screenshot_path),
                }
            )

        response = client.responses.create(
            model=cfg.model_name,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": prompt}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )

        raw_output_text = _extract_text(response)
        parsed = _parse_payload(raw_output_text)

        brand_guess = str(parsed.get("brand_guess", "unknown") or "unknown")
        task_guess = str(parsed.get("task_guess", "unknown") or "unknown")
        reasons_val = parsed.get("reasons", [])
        reasons = (
            [str(item) for item in reasons_val]
            if isinstance(reasons_val, list)
            else [str(reasons_val)]
        )
        unknown = bool(parsed.get("unknown", True))

        normalized_task = normalize_task_guess(
            task_guess,
            password_field_present=password_field_present,
            final_url=final_url,
            visible_text=visible_text,
        )
        if normalized_task != task_guess:
            reasons = list(reasons)
            reasons.append(
                "Task was normalized to homepage because no password field appeared in the "
                "captured viewport (sign-in links alone are not treated as a login-primary page)."
            )
            task_guess = normalized_task
        if url_intel and brand_guess == "unknown" and url_intel.brand_hint != "unknown":
            brand_guess = url_intel.brand_hint
            reasons = list(reasons)
            reasons.append("Brand guess backfilled from deterministic URL intelligence hint.")
        if url_intel and task_guess in {"unknown", ""} and url_intel.action_hint not in {"unknown", ""}:
            task_guess = url_intel.action_hint
            reasons = list(reasons)
            reasons.append("Task guess backfilled from deterministic URL intelligence hint.")
        if not screenshot_used:
            reasons = list(reasons)
            reasons.append(
                "Screenshot capture was unavailable, so AI used text and URL signals only."
            )

        return AIBrandTaskResult(
            brand_guess=brand_guess,
            task_guess=task_guess,
            reasons=reasons,
            unknown=unknown,
            raw_output_text=raw_output_text,
        )
    except Exception as exc:  # noqa: BLE001 - Preserve pipeline continuity.
        return AIBrandTaskResult(error=f"ai_inference_failed: {exc}")


def build_intent_summary(
    *,
    brand_guess: str,
    task_guess: str,
    final_url: str,
    visible_text: str,
) -> str:
    """Produce analyst-friendly one-line intent summary for suspicious URL."""
    brand = (brand_guess or "unknown").strip()
    task = (task_guess or "unknown").strip().lower()
    url_hint = (final_url or "").lower()
    text_hint = (visible_text or "").lower()

    if brand.lower() == "unknown":
        if any(t in url_hint or t in text_hint for t in ("login", "signin", "verify", "password")):
            return "This URL appears to be presenting an authentication-oriented flow."
        if any(t in url_hint or t in text_hint for t in ("checkout", "payment", "invoice")):
            return "This URL appears to be presenting a payment or checkout flow."
        return "This URL appears to be attempting a brand-associated web flow, but the exact brand is unclear."

    if task == "login":
        return f"This URL appears to be trying to imitate a {brand} login page."
    if task == "checkout":
        return f"This URL appears to be a {brand} checkout/payment page."
    if task == "password reset":
        return f"This URL appears to be a {brand} password reset flow."
    if task == "account verification":
        return f"This URL appears to be a {brand} account verification flow."
    if task == "document share":
        return f"This URL appears to be a {brand} document-share access flow."
    if task in {"homepage", "browse"}:
        return f"This URL appears to imitate a {brand} homepage/browsing experience."
    return f"This URL appears to be a {brand} {task} flow."
