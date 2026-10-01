"""The in-browser demo (demo/js/phishguard-engine.js) must give the same answers as the Python app.

Rebuild Phase 6. Re-exports the shipped model, writes the Python reference for a fixed URL set
(demo/parity/fixture_urls.jsonl: 300 seeded held-out test URLs plus every curated evaluation URL and
the port edge cases) with the real ML-only dashboard, then runs the JS engine on the same URLs.
Features, verdicts, votes and host checks must match exactly; probabilities within 1e-6.
The full 147,714-URL run is documented in demo/parity/results/.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_js_engine_matches_python(tmp_path):
    env = dict(os.environ)
    build = tmp_path / "build"
    subprocess.run([sys.executable, str(REPO / "demo" / "export_model.py"), "--out", str(build)], check=True, env=env, cwd=REPO)
    ref = tmp_path / "reference.jsonl"
    subprocess.run(
        [sys.executable, str(REPO / "demo" / "parity" / "make_reference.py"), "--urls", str(REPO / "demo" / "parity" / "fixture_urls.jsonl"),
         "--out", str(ref), "--dashboard"],
        check=True, env=env, cwd=REPO,
    )
    summary = tmp_path / "summary.json"
    proc = subprocess.run(
        [NODE, str(REPO / "demo" / "parity" / "run_parity.js"), str(REPO / "demo" / "parity" / "fixture_urls.jsonl"), str(ref), str(summary), str(build)],
        cwd=REPO, capture_output=True, text=True,
    )
    s = json.loads(summary.read_text())
    assert s["rows"] >= 1300 and s["dashboard_rows"] == s["rows"]
    assert proc.returncode == 0 and s["pass"], json.dumps(s["examples"], ensure_ascii=False)[:4000]
