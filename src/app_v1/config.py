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
    post_load_stabilize_ms: int = 750
    # After optional login-form submit, wait before post-submit viewport screenshot.
    post_submit_stabilize_ms: int = 1200
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
            wait_until=os.getenv("PHISH_WAIT_UNTIL", cls.wait_until),
        )

    def as_json(self) -> Dict[str, Any]:
        """Return JSON-serializable config dict."""
        return asdict(self)

    def ensure_output_dir(self) -> Path:
        """Ensure output directory exists and return it."""
        out_path = Path(self.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path
