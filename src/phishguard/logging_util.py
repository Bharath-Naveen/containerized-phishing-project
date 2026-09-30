"""Central logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
