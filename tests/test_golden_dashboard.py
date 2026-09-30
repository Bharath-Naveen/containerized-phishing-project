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
    return importlib.import_module("phishguard.app.dashboard")


@pytest.mark.parametrize("case", _data["cases"], ids=lambda c: c.get("name") or c["url"][:60])
def test_dashboard_output_unchanged(case):
    dash = _dashboard_module()
    capture = _rec.make_capture(_rec.CAPTURE_CASES[case["name"]]) if case["kind"] == "capture" else None
    got = _rec.run_case(dash, case["url"], case["ml"], case["agreement"], capture)
    diffs = _diff(case["expected"], got)
    assert not diffs, "golden mismatch:\n" + "\n".join(diffs[:15])


def _diff(a, b, path="") -> list:
    if type(a) is not type(b):
        return [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'expected' if k in a else 'got'}")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: list len {len(a)} != {len(b)} ({a!r} vs {b!r})"[:400]]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"[:400]]


def test_golden_file_present():
    assert len(_data["cases"]) >= 100
