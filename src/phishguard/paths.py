"""Filesystem layout and environment-driven roots."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Repo root: parent of ``src/``."""
    env = os.environ.get("PHISH_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("PHISH_DATA_DIR", project_root() / "data")).resolve()


def outputs_dir() -> Path:
    return Path(os.environ.get("PHISH_OUTPUTS_DIR", project_root() / "outputs")).resolve()


def logs_dir() -> Path:
    return Path(os.environ.get("PHISH_LOGS_DIR", project_root() / "logs")).resolve()


def raw_dir() -> Path:
    return data_dir() / "raw"


def interim_dir() -> Path:
    return data_dir() / "interim"


def processed_dir() -> Path:
    return data_dir() / "processed"


def models_dir() -> Path:
    return outputs_dir() / "models"


def metrics_dir() -> Path:
    return outputs_dir() / "metrics"


def reports_dir() -> Path:
    return outputs_dir() / "reports"


def figures_dir() -> Path:
    return outputs_dir() / "figures"


def analysis_dir() -> Path:
    return outputs_dir() / "analysis"


def ensure_layout() -> None:
    for p in (
        raw_dir(),
        interim_dir(),
        processed_dir(),
        models_dir(),
        metrics_dir(),
        reports_dir(),
        figures_dir(),
        analysis_dir(),
        logs_dir(),
    ):
        p.mkdir(parents=True, exist_ok=True)
