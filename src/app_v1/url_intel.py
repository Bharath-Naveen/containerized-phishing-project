"""URL-intelligence heuristics and lightweight plausibility checks."""

from __future__ import annotations

import os
from urllib.parse import parse_qs, urlparse

from openai import OpenAI

from .config import PipelineConfig
from .schemas import URLIntelResult


def _infer_language_and_locale(path: str, query_keys: list[str], query_map: dict[str, list[str]]) -> tuple[str | None, str | None]:
    low = (path or "").lower()
    for lang in ("fr", "de", "es", "en", "pt"):
        if f"/{lang}/" in low or low.startswith(f"/{lang}/"):
            locale = {"en": "en-US", "fr": "fr-FR", "de": "de-DE", "es": "es-ES", "pt": "pt-BR"}[lang]
            return lang, locale
    for key in query_keys:
        if key in {"lang", "locale", "hl", "gl"}:
            val = (query_map.get(key, [""])[0] or "").lower()
            if len(val) >= 2:
                lang = val[:2]
                locale = {"en": "en-US", "fr": "fr-FR", "de": "de-DE", "es": "es-ES", "pt": "pt-BR"}.get(lang)
                return lang, locale
    return None, None


def _heuristic_hints(hostname: str, path: str, fragment: str, query_keys: list[str]) -> tuple[str, str, str, list[str]]:
    host = (hostname or "").lower()
    p = (path or "").lower()
    f = (fragment or "").lower()
    blob = " ".join([host, p, f, " ".join(query_keys)])
    reasons: list[str] = []
    brand = "unknown"
    product = "unknown"
    action = "unknown"

    if "google" in host:
        brand = "google"
        if "lens.google" in host or "homework" in f or "visualsearch" in blob:
            product = "google_lens"; reasons.append("URL indicates Google Lens surface")
        elif "drive.google" in host:
            product = "google_drive"; reasons.append("URL indicates Google Drive surface")
        elif "mail.google" in host or "gmail" in blob:
            product = "google_gmail"; reasons.append("URL indicates Gmail surface")
        elif "photos.google" in host:
            product = "google_photos"; reasons.append("URL indicates Google Photos surface")
        elif "workspace.google.com" in host and "drive" in blob:
            product = "google_drive"; reasons.append("URL indicates Google Workspace Drive surface")
    elif "microsoft" in host or "office.com" in host or "sharepoint.com" in host or "live.com" in host:
        brand = "microsoft"
        if "outlook" in blob:
            product = "microsoft_outlook"; reasons.append("URL indicates Microsoft Outlook surface")
        elif "onedrive" in blob:
            product = "microsoft_onedrive"; reasons.append("URL indicates Microsoft OneDrive surface")
        elif "sharepoint" in blob:
            product = "microsoft_sharepoint"; reasons.append("URL indicates Microsoft SharePoint surface")
    elif "paypal" in host:
        brand = "paypal"; product = "paypal_web"
    elif "amazon" in host:
        brand = "amazon"; product = "amazon_web"
    elif "appleid" in host or "apple" in host:
        brand = "apple"; product = "apple_id" if "appleid." in host else "apple_web"
    elif "facebook" in host:
        brand = "facebook"; product = "facebook_web"
    elif "stripe" in host:
        brand = "stripe"; product = "stripe_checkout"

    if any(t in blob for t in ("login", "signin", "sign-in", "auth")):
        action = "login"
    elif any(t in blob for t in ("checkout", "payment", "invoice")):
        action = "checkout"
    elif any(t in blob for t in ("returns", "orders", "track", "shipment", "delivery")):
        action = "informational"
    elif any(t in blob for t in ("reset", "recover", "verify", "confirmation")):
        action = "account verification"
    elif any(t in blob for t in ("suspend", "billing issue", "storage", "urgent")):
        action = "informational"
    elif any(t in blob for t in ("home", "homepage")):
        action = "homepage"
    return brand, product, action, reasons


def _heuristic_plausibility(hostname: str, brand_hint: str, product_hint: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    host = (hostname or "").lower()
    plausible = "unknown"
    if brand_hint != "unknown":
        if brand_hint in {"google", "microsoft", "paypal", "amazon", "apple", "facebook", "stripe"} and brand_hint not in host:
            plausible = "suspicious_shape"
            reasons.append("Brand hint and hostname do not align")
        else:
            plausible = "likely_first_party"
            reasons.append("Hostname appears consistent with inferred brand/product")
    if product_hint in {"google_lens"} and "lens.google" not in host:
        plausible = "suspicious_shape"
        reasons.append("Product hint implies lens.google but hostname differs")
    return plausible, reasons


def _ai_url_shape_check(url: str, hostname: str, path: str, brand_hint: str, product_hint: str, cfg: PipelineConfig) -> tuple[str, list[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "unknown", []
    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Classify URL first-party plausibility. Return strict JSON with keys "
            "first_party_url_plausibility (likely_first_party|suspicious_shape|unknown) and url_shape_reasons (array)."
        )
        user = f"url={url}\nhost={hostname}\npath={path}\nbrand_hint={brand_hint}\nproduct_hint={product_hint}"
        r = client.responses.create(
            model=cfg.model_name,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
        )
        txt = getattr(r, "output_text", "") or ""
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            import json

            obj = json.loads(txt[start : end + 1])
            p = str(obj.get("first_party_url_plausibility", "unknown"))
            rs = obj.get("url_shape_reasons", [])
            return p if p in {"likely_first_party", "suspicious_shape", "unknown"} else "unknown", [str(x) for x in rs] if isinstance(rs, list) else []
    except Exception:
        pass
    return "unknown", []


def analyze_url_intel(input_url: str, config: PipelineConfig | None = None) -> URLIntelResult:
    cfg = config or PipelineConfig.from_env()
    try:
        parsed = urlparse((input_url or "").strip())
        normalized = parsed.geturl()
        query_map = parse_qs(parsed.query or "", keep_blank_values=False)
        query_keys = sorted(query_map.keys())
        brand, product, action, reasons = _heuristic_hints(
            parsed.hostname or "", parsed.path or "", parsed.fragment or "", query_keys
        )
        lang, locale = _infer_language_and_locale(parsed.path or "", query_keys, query_map)
        plaus, plaus_reasons = _heuristic_plausibility(parsed.hostname or "", brand, product)
        ai_plaus, ai_reasons = _ai_url_shape_check(
            normalized, parsed.hostname or "", parsed.path or "", brand, product, cfg
        )
        final_plaus = ai_plaus if ai_plaus != "unknown" else plaus
        all_reasons = reasons + plaus_reasons + ai_reasons
        return URLIntelResult(
            normalized_url=normalized,
            hostname=(parsed.hostname or "").lower(),
            path=parsed.path or "",
            fragment=parsed.fragment or "",
            query_keys=query_keys,
            brand_hint=brand,
            product_hint=product,
            action_hint=action,
            language_hint=lang,
            locale_hint=locale,
            first_party_url_plausibility=final_plaus,
            url_shape_reasons=all_reasons,
        )
    except Exception as exc:  # noqa: BLE001
        return URLIntelResult(error=f"url_intel_failed: {type(exc).__name__}: {exc}")

