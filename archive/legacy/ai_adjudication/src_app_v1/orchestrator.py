"""End-to-end orchestrator for phishing triage dataset row creation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .ai_brand_task import build_intent_summary, infer_brand_and_task
from .capture import capture_url
from .compare import compare_suspicious_vs_legit_reference
from .config import PipelineConfig
from .feature_extract import extract_features
from .legit_lookup import lookup_legitimate_urls
from .schemas import (
    AIBrandTaskResult,
    CaptureResult,
    ComparisonResult,
    DatasetRow,
    FeatureResult,
    LegitLookupResult,
    URLIntelResult,
    VerdictResult,
    utc_now_iso,
)
from .url_intel import analyze_url_intel
from .verdict import generate_verdict

logger = logging.getLogger(__name__)


def run_pipeline(url: str, config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """Run full pipeline and always return a JSON-serializable row.

    Each ``capture`` / ``legit_reference_capture`` dict includes ``screenshot_path``
    (viewport), ``fullpage_screenshot_path``, ``html_path``, ``interaction`` (dummy
    login probe metadata), and related fields from
    :class:`~app_v1.schemas.CaptureResult`.

    The ``comparison`` object includes deterministic alignment between suspicious and
    legit reference captures (titles, visible text, password/login signals, post-submit
    host vs trusted domain, task alignment) via :class:`~app_v1.schemas.ComparisonResult`.
    """
    cfg = config or PipelineConfig.from_env()
    pipeline_errors: list[str] = []

    # Initialize every stage with safe defaults so we always return a stable row shape.
    capture = CaptureResult(
        original_url=url,
        final_url=url,
        title="",
        screenshot_path="",
        fullpage_screenshot_path="",
        html_path="",
        visible_text="",
        error="capture_not_started",
        capture_blocked=False,
        capture_strategy="playwright_headless",
        capture_block_reason=None,
        capture_block_evidence=None,
        first_failed_capture_step=None,
    )
    ai = AIBrandTaskResult(error="ai_not_started")
    url_intel = URLIntelResult(error="url_intel_not_started")
    legit = LegitLookupResult(error="legit_lookup_not_started")
    legit_reference_capture: Optional[CaptureResult] = None
    features = FeatureResult(
        input_url=url,
        final_url=url,
        final_domain="",
        has_form=False,
        external_link_ratio=0.0,
        title_length=0,
        visible_text_length=0,
        capture_blocked=False,
        error="feature_extract_not_started",
    )
    comparison = ComparisonResult(
        brand_guess="unknown",
        task_guess="unknown",
        trusted_reference_found=False,
        reasons=[],
        error="comparison_not_started",
    )
    verdict = VerdictResult(
        verdict="inconclusive",
        confidence="low",
        reasons=["Verdict could not be fully evaluated due to earlier stage failures."],
        error="verdict_not_started",
    )

    # Capture suspicious URL evidence.
    try:
        capture = capture_url(url, cfg, namespace="suspicious")
        if capture.error:
            pipeline_errors.append(f"capture: {capture.error}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pipeline stage=capture failed for url=%s", url)
        capture.error = f"capture_exception: {type(exc).__name__}: {exc}"
        pipeline_errors.append(f"capture_exception: {exc}")

    # URL intelligence.
    try:
        url_intel = analyze_url_intel(url, cfg)
        if url_intel.error:
            pipeline_errors.append(f"url_intel: {url_intel.error}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pipeline stage=url_intel failed for url=%s", url)
        url_intel.error = f"url_intel_exception: {type(exc).__name__}: {exc}"
        pipeline_errors.append(f"url_intel_exception: {exc}")

    # AI brand + task inference.
    try:
        ai = infer_brand_and_task(
            screenshot_path=capture.screenshot_path,
            visible_text=capture.visible_text,
            title=capture.title,
            final_url=capture.final_url,
            config=cfg,
            password_field_present=capture.interaction.password_field_found,
            url_intel=url_intel,
        )
        if ai.error:
            pipeline_errors.append(f"ai_brand_task: {ai.error}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pipeline stage=ai_brand_task failed for url=%s", url)
        ai.error = f"ai_brand_task_exception: {type(exc).__name__}: {exc}"
        pipeline_errors.append(f"ai_brand_task_exception: {exc}")

    # Trusted reference lookup.
    try:
        legit = lookup_legitimate_urls(
            ai.brand_guess,
            ai.task_guess,
            input_url=url,
            product_hint=url_intel.product_hint,
            action_hint=url_intel.action_hint,
            language_hint=url_intel.language_hint,
        )
        if legit.error:
            pipeline_errors.append(f"legit_lookup: {legit.error}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pipeline stage=legit_lookup failed for url=%s", url)
        legit.error = f"legit_lookup_exception: {type(exc).__name__}: {exc}"
        pipeline_errors.append(f"legit_lookup_exception: {exc}")

    # Capture trusted reference page when available.
    legit_reference_url: Optional[str] = legit.candidate_urls[0] if legit.candidate_urls else None
    if legit_reference_url:
        try:
            legit_reference_capture = capture_url(
                legit_reference_url,
                cfg,
                namespace="legit",
                preferred_language=(
                    url_intel.language_hint or capture.detected_language
                ),
            )
            if legit_reference_capture.error:
                pipeline_errors.append(
                    f"legit_reference_capture: {legit_reference_capture.error}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "run_pipeline stage=legit_reference_capture failed for url=%s ref=%s",
                url,
                legit_reference_url,
            )
            pipeline_errors.append(f"legit_reference_capture_exception: {exc}")
            legit_reference_capture = None

    # Feature extraction.
    try:
        features = extract_features(
            input_url=url,
            final_url=capture.final_url,
            title=capture.title,
            visible_text=capture.visible_text,
            html_path=capture.html_path,
            capture_blocked=capture.capture_blocked,
            capture_block_reason=capture.capture_block_reason,
            capture_block_evidence=capture.capture_block_evidence,
            screenshot_path=capture.screenshot_path,
            redirect_count=capture.redirect_count,
            redirect_chain=capture.redirect_chain,
            cross_domain_redirect_count=capture.cross_domain_redirect_count,
            settled_successfully=capture.settled_successfully,
            settle_time_ms=capture.settle_time_ms,
            suspicious_language=capture.detected_language,
            legit_language=(
                legit_reference_capture.detected_language
                if legit_reference_capture
                else None
            ),
            task_guess=ai.task_guess,
            legit_title=(
                legit_reference_capture.title if legit_reference_capture else ""
            ),
            legit_visible_text=(
                legit_reference_capture.visible_text
                if legit_reference_capture
                else ""
            ),
            legit_reference_match_tier=legit.legit_reference_match_tier,
            url_brand_hint=url_intel.brand_hint,
            url_product_hint=url_intel.product_hint,
            url_action_hint=url_intel.action_hint,
            url_language_hint=url_intel.language_hint,
            url_first_party_plausibility=url_intel.first_party_url_plausibility,
            url_shape_reasons=url_intel.url_shape_reasons,
        )
        if features.error:
            pipeline_errors.append(f"feature_extract: {features.error}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pipeline stage=feature_extract failed for url=%s", url)
        features.error = f"feature_extract_exception: {type(exc).__name__}: {exc}"
        pipeline_errors.append(f"feature_extract_exception: {exc}")

    # Capture comparison.
    try:
        comparison = compare_suspicious_vs_legit_reference(
            suspicious=capture,
            legit=legit_reference_capture,
            brand_guess=ai.brand_guess,
            task_guess=ai.task_guess,
            trusted_reference_found=legit.matched,
            matched_legit_urls=legit.candidate_urls,
            url_product_hint=url_intel.product_hint,
            url_action_hint=url_intel.action_hint,
        )
        if comparison.error:
            pipeline_errors.append(f"comparison: {comparison.error}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pipeline stage=comparison failed for url=%s", url)
        comparison.error = f"comparison_exception: {type(exc).__name__}: {exc}"
        comparison.reasons.append(
            "Comparison stage failed; verdict used available deterministic signals only."
        )
        pipeline_errors.append(f"comparison_exception: {exc}")

    # Final verdict.
    try:
        verdict = generate_verdict(
            ai_unknown=ai.unknown,
            comparison=comparison,
            features=features,
        )
        if verdict.error:
            pipeline_errors.append(f"verdict: {verdict.error}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("run_pipeline stage=verdict failed for url=%s", url)
        verdict.error = f"verdict_exception: {type(exc).__name__}: {exc}"
        verdict.verdict = "inconclusive"
        verdict.confidence = "low"
        verdict.reasons = [
            "Verdict stage failed unexpectedly; earlier stage results are still included."
        ]
        pipeline_errors.append(f"verdict_exception: {exc}")

    pipeline_error = " | ".join(pipeline_errors) if pipeline_errors else None
    capture.intent_summary = build_intent_summary(
        brand_guess=ai.brand_guess,
        task_guess=ai.task_guess,
        final_url=capture.final_url,
        visible_text=capture.visible_text,
    )

    row = DatasetRow(
        timestamp_utc=utc_now_iso(),
        input_url=url,
        url_intel=url_intel.as_json(),
        capture=capture.as_json(),
        ai_brand_task=ai.as_json(),
        legit_lookup=legit.as_json(),
        legit_reference_capture=legit_reference_capture.as_json() if legit_reference_capture else None,
        features=features.as_json(),
        comparison=comparison.as_json(),
        verdict=verdict.as_json(),
        error=pipeline_error,
    )
    return row.as_json()


def save_dataset_row(row: Dict[str, Any], output_file: str) -> str:
    """Append JSON row to a JSONL file for dataset building."""
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(row, ensure_ascii=False)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(serialized + "\n")
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phishing triage pipeline.")
    parser.add_argument("--url", required=True, help="URL to triage")
    parser.add_argument(
        "--out",
        default="data/triage_rows_v1.jsonl",
        help="JSONL output path for dataset rows",
    )
    args = parser.parse_args()

    row = run_pipeline(args.url)
    path = save_dataset_row(row, args.out)
    print(json.dumps({"saved_to": path, "row": row}, indent=2))


if __name__ == "__main__":
    main()
