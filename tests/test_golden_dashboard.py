"""Golden-output test: the dashboard must return exactly what it returned before a refactor.

See tests/golden/record_golden.py for what is frozen and how to re-record (only for intended changes).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).resolve().parent / "golden" / "dashboard_golden.json"
_rec = importlib.import_module("tests.golden.record_golden")
_data = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.is_file() else {"cases": []}


def _dashboard_module():
    for name in ("src.phishguard.app.dashboard", "src.app_v1.analyze_dashboard"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    raise ImportError("dashboard module not found")


@pytest.mark.parametrize("case", _data["cases"], ids=lambda c: c.get("name") or c["url"][:60])
def test_dashboard_output_unchanged(case):
    dash = _dashboard_module()
    capture = _rec.make_capture(_rec.CAPTURE_CASES[case["name"]]) if case["kind"] == "capture" else None
    got = _rec.run_case(dash, case["url"], case["ml"], case["agreement"], capture)
    assert got == case["expected"]


def test_golden_file_present():
    assert len(_data["cases"]) >= 100
