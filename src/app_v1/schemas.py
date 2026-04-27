"""Shared schemas and JSON-safe conversion helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class CaptureInteractionMetadata:
    """Optional Playwright login-form probe (dummy credentials, non-fatal)."""

    attempted_submit: bool = False
    user_field_found: bool = False
    password_field_found: bool = False
    submit_found_and_clicked: bool = False
    url_before_submit: Optional[str] = None
    url_after_submit: Optional[str] = None
    navigation_occurred: bool = False
    post_submit_viewport_screenshot_path: Optional[str] = None
    interaction_error: Optional[str] = None
    click_probe_attempted: bool = False
    click_probe_text: Optional[str] = None
    click_probe_before_url: Optional[str] = None
    click_probe_after_url: Optional[str] = None
    click_probe_domain_changed: Optional[bool] = None
    click_probe_error: Optional[str] = None
    click_probe_enabled: bool = False
    click_probe_candidate_count: int = 0
    click_probe_candidate_texts_sample: List[str] = field(default_factory=list)
    click_probe_skip_reason: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaptureResult:
    original_url: str
    final_url: str
    title: str
    # Viewport (mobile) screenshot — what fits in the fixed viewport.
    screenshot_path: str
    # Full scrollable page screenshot (may be tall).
    fullpage_screenshot_path: str
    html_path: str
    visible_text: str
    initial_url: str = ""
    redirect_chain: List[str] = field(default_factory=list)
    redirect_count: int = 0
    cross_domain_redirect_count: int = 0
    settle_time_ms: int = 0
    settled_successfully: bool = False
    detected_language: Optional[str] = None
    language_source: Optional[str] = None
    intent_summary: Optional[str] = None
    interaction: CaptureInteractionMetadata = field(
        default_factory=CaptureInteractionMetadata
    )
    error: Optional[str] = None
    # True if the target resisted automation (blocked navigation) and/or only HTTP HTML was retrieved.
    capture_blocked: bool = False
    # How capture completed: playwright_headless | playwright_stealth | http_fallback | failed
    capture_strategy: str = "playwright_headless"
    # Coarse category for capture blocking behavior (if blocked/evasive behavior observed).
    capture_block_reason: Optional[str] = None
    # Short human-readable explanation with concrete evidence from capture attempts.
    capture_block_evidence: Optional[str] = None
    # First Playwright/sync sub-step that failed (chronological), if any; e.g. "title extraction (page.title)".
    first_failed_capture_step: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        # asdict recurses into nested dataclasses (e.g. interaction).
        return asdict(self)


@dataclass
class AIBrandTaskResult:
    brand_guess: str = "unknown"
    task_guess: str = "unknown"
    reasons: List[str] = field(default_factory=list)
    unknown: bool = True
    raw_output_text: str = ""
    error: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class URLIntelResult:
    normalized_url: str = ""
    hostname: str = ""
    path: str = ""
    fragment: str = ""
    query_keys: List[str] = field(default_factory=list)
    brand_hint: str = "unknown"
    product_hint: str = "unknown"
    action_hint: str = "unknown"
    language_hint: Optional[str] = None
    locale_hint: Optional[str] = None
    first_party_url_plausibility: str = "unknown"  # likely_first_party | suspicious_shape | unknown
    url_shape_reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LegitLookupResult:
    candidate_urls: List[str] = field(default_factory=list)
    matched: bool = False
    selected_reference_url: Optional[str] = None
    legit_reference_match_tier: str = "unknown"
    error: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureResult:
    input_url: str
    final_url: str
    final_domain: str
    has_form: bool
    external_link_ratio: float
    title_length: int
    visible_text_length: int
    capture_blocked: bool = False
    capture_block_reason: Optional[str] = None
    capture_block_evidence: Optional[str] = None
    visual_capture_unavailable: bool = False
    url_contains_auth_tokens: bool = False
    brand_lookalike_signal: bool = False
    brand_lookalike_to: Optional[str] = None
    trusted_brand_root_mismatch: bool = False
    redirect_count: int = 0
    redirect_chain: List[str] = field(default_factory=list)
    cross_domain_redirect_count: int = 0
    settled_successfully: bool = False
    settle_time_ms: int = 0
    suspicious_language: Optional[str] = None
    legit_language: Optional[str] = None
    language_match: Optional[bool] = None
    legit_reference_matches_intended_task: Optional[bool] = None
    legit_reference_quality: str = "unknown"
    legit_reference_match_tier: str = "unknown"
    url_brand_hint: str = "unknown"
    url_product_hint: str = "unknown"
    url_action_hint: str = "unknown"
    url_language_hint: Optional[str] = None
    url_first_party_plausibility: str = "unknown"
    url_shape_reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Structured comparison of suspicious capture vs trusted legitimate reference capture."""

    # Inputs / classifier output (duplicated here for convenient row-level analysis).
    brand_guess: str
    task_guess: str

    # Trusted-reference matching derived from (brand_guess, task_guess).
    trusted_reference_found: bool
    matched_legit_urls: List[str] = field(default_factory=list)

    # --- Suspicious vs legit capture alignment (deterministic, no CV) ---
    title_similarity: float = 0.0  # 0..1 (SequenceMatcher on titles)
    visible_text_similarity: float = 0.0  # 0..1 on truncated visible text
    suspicious_password_field_present: bool = False
    legit_password_field_present: bool = False
    suspicious_login_interaction_possible: bool = False  # user + password fields detected
    legit_login_interaction_possible: bool = False
    # None = not evaluated. True = post-submit moved off trusted ref in a *suspicious* way
    # (unknown host; excludes trusted IdP / common OAuth URL patterns).
    post_submit_left_trusted_domain: Optional[bool] = None
    # True if post-submit URL matches OAuth heuristics (oauth, redirect_uri, accounts.google.com, etc.).
    is_oauth_flow: bool = False
    # True if post-submit host is a known federated login / IdP domain (Google, Apple, Microsoft).
    trusted_oauth_redirect: bool = False
    task_aligned_with_legit_reference: bool = False
    language_match: Optional[bool] = None
    legit_reference_matches_intended_task: Optional[bool] = None
    legit_reference_quality: str = "unknown"
    legit_reference_match_tier: str = "unknown"
    url_product_action_aligned: Optional[bool] = None

    # Similarity / gap scores (legacy + derived behavior gap).
    action_match_score: float = 0.0
    visual_similarity_score: float = 0.0  # TODO: pending image-based similarity implementation.
    dom_similarity_score: float = 0.0  # HTML tag / form heuristic vs reference
    behavior_gap_score: float = 0.0  # derived from structured alignment signals

    # Plain-English and short diagnostic strings for analysts / models.
    reasons: List[str] = field(default_factory=list)
    # Populated only when comparison computation itself fails.
    error: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerdictResult:
    verdict: str
    confidence: str
    reasons: List[str]
    error: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetRow:
    timestamp_utc: str
    input_url: str
    url_intel: Dict[str, Any]
    capture: Dict[str, Any]
    ai_brand_task: Dict[str, Any]
    legit_lookup: Dict[str, Any]
    # Required field (can be None) to keep dataclass field ordering valid.
    legit_reference_capture: Optional[Dict[str, Any]]
    features: Dict[str, Any]
    comparison: Dict[str, Any]
    verdict: Dict[str, Any]
    error: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)
