"""Test isolation (rebuild Phase 1).

Tests used to write reports and CSVs into the real outputs/ and data/processed/ folders
(for example outputs/reports/split_leak_safe_stats.json was overwritten by a test). Every test
session now runs with PHISH_OUTPUTS_DIR / PHISH_DATA_DIR / PHISH_LOGS_DIR pointed at a temp
folder. Shipped models are copied into the temp outputs so model smoke tests still run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _isolated_output_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("phish_env")
    out, data, logs = root / "outputs", root / "data", root / "logs"
    (out / "models").mkdir(parents=True)
    (out / "reports").mkdir(parents=True)
    real_out = Path(os.environ.get("PHISH_OUTPUTS_DIR", REPO / "outputs"))
    for p in (real_out / "models").glob("*.joblib"):
        if p.stat().st_size > 16 and "_2026" not in p.name:
            shutil.copy(p, out / "models" / p.name)
    tc = real_out / "reports" / "training_config.json"
    if tc.is_file():
        shutil.copy(tc, out / "reports" / tc.name)
    saved = {k: os.environ.get(k) for k in ("PHISH_OUTPUTS_DIR", "PHISH_DATA_DIR", "PHISH_LOGS_DIR")}
    os.environ["PHISH_OUTPUTS_DIR"] = str(out)
    os.environ["PHISH_DATA_DIR"] = str(data)
    os.environ["PHISH_LOGS_DIR"] = str(logs)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
