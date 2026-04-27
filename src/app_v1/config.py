"""Configuration for the phishing triage pipeline."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the triage pipeline."""

    output_dir: str = "captures/app_v1"
    model_name: str = "gpt-4.1"
    navigation_timeout_ms: int = 30000
    # After navigation, wait briefly so JS-heavy pages can render before capture.
    post_load_stabilize_ms: int = 3000
    # After optional login-form submit, wait before post-submit viewport screenshot.
    post_submit_stabilize_ms: int = 3000
    # Optional safe click probe (no typing) to reveal post-click redirects.
    enable_click_probe: bool = False
    wait_until: str = "domcontentloaded"
    fixed_viewport_width: int = 390
    fixed_viewport_height: int = 844
    user_agent: str = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 "
        "Mobile/15E148 Safari/604.1"
    )
    visible_text_max_chars: int = 10000
    ai_text_max_chars: int = 4000
    legitimacy_rescue_enabled: bool = True
    legitimacy_rescue_max_html_structure_risk: float = 0.32
    legitimacy_rescue_max_html_dom_anomaly_risk: float = 0.30
    legitimacy_rescue_ml_cap_after_rescue: float = 0.52
    legitimacy_rescue_target_verdict: str = "uncertain"
    trusted_domains_csv_path: str = "data/reference/trusted_domains.csv"
    platform_domains_csv_path: str = "data/reference/platform_domains.csv"
    legitimacy_rescue_dns_dampening_factor: float = 0.78
    legitimacy_rescue_dns_contribution_threshold: float = 0.35

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load config from environment with sane defaults."""
        return cls(
            output_dir=os.getenv("PHISH_OUTPUT_DIR", cls.output_dir),
            model_name=os.getenv("PHISH_OPENAI_MODEL", cls.model_name),
            navigation_timeout_ms=int(
                os.getenv("PHISH_NAV_TIMEOUT_MS", str(cls.navigation_timeout_ms))
            ),
            post_load_stabilize_ms=int(
                os.getenv("PHISH_POST_LOAD_STABILIZE_MS", str(cls.post_load_stabilize_ms))
            ),
            post_submit_stabilize_ms=int(
                os.getenv("PHISH_POST_SUBMIT_STABILIZE_MS", str(cls.post_submit_stabilize_ms))
            ),
            enable_click_probe=os.getenv("PHISH_ENABLE_CLICK_PROBE", "false").strip().lower() in {"1", "true", "yes", "on"},
            wait_until=os.getenv("PHISH_WAIT_UNTIL", cls.wait_until),
            legitimacy_rescue_enabled=os.getenv("PHISH_LEGITIMACY_RESCUE_ENABLED", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            legitimacy_rescue_max_html_structure_risk=float(
                os.getenv(
                    "PHISH_LEGITIMACY_RESCUE_MAX_HTML_STRUCTURE_RISK",
                    str(cls.legitimacy_rescue_max_html_structure_risk),
                )
            ),
            legitimacy_rescue_max_html_dom_anomaly_risk=float(
                os.getenv(
                    "PHISH_LEGITIMACY_RESCUE_MAX_HTML_DOM_ANOMALY_RISK",
                    str(cls.legitimacy_rescue_max_html_dom_anomaly_risk),
                )
            ),
            legitimacy_rescue_ml_cap_after_rescue=float(
                os.getenv(
                    "PHISH_LEGITIMACY_RESCUE_ML_CAP_AFTER_RESCUE",
                    str(cls.legitimacy_rescue_ml_cap_after_rescue),
                )
            ),
            legitimacy_rescue_target_verdict=os.getenv(
                "PHISH_LEGITIMACY_RESCUE_TARGET_VERDICT",
                cls.legitimacy_rescue_target_verdict,
            ).strip().lower(),
            trusted_domains_csv_path=os.getenv(
                "PHISH_TRUSTED_DOMAINS_CSV_PATH",
                cls.trusted_domains_csv_path,
            ),
            platform_domains_csv_path=os.getenv(
                "PHISH_PLATFORM_DOMAINS_CSV_PATH",
                cls.platform_domains_csv_path,
            ),
            legitimacy_rescue_dns_dampening_factor=float(
                os.getenv(
                    "PHISH_LEGITIMACY_RESCUE_DNS_DAMPENING_FACTOR",
                    str(cls.legitimacy_rescue_dns_dampening_factor),
                )
            ),
            legitimacy_rescue_dns_contribution_threshold=float(
                os.getenv(
                    "PHISH_LEGITIMACY_RESCUE_DNS_CONTRIBUTION_THRESHOLD",
                    str(cls.legitimacy_rescue_dns_contribution_threshold),
                )
            ),
        )

    def as_json(self) -> Dict[str, Any]:
        """Return JSON-serializable config dict."""
        return asdict(self)

    def ensure_output_dir(self) -> Path:
        """Ensure output directory exists and return it."""
        out_path = Path(self.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path
