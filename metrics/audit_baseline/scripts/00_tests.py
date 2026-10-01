"""Step 0: run the pytest suite and record collected / passed / failed / skipped and line coverage.

The suite is run with PHISH_OUTPUTS_DIR / PHISH_DATA_DIR pointed at metrics/_work so tests that
write reports (e.g. phish_audit, split stats) do not overwrite files in outputs/ or data/processed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from common import REPO, WORK, isolate_env, write_result

isolate_env()


def main() -> None:
    junit = WORK / "junit.xml"
    cov = WORK / "coverage.json"
    cmd = [
        sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
        "--cov=src", f"--cov-report=json:{cov}", f"--junitxml={junit}",
    ]
    # Give the model smoke tests a copy of the shipped models, in an isolated outputs dir.
    import shutil

    test_out = WORK / "test_outputs"
    if test_out.exists():
        shutil.rmtree(test_out)
    (test_out / "models").mkdir(parents=True)
    for p in (REPO / "outputs" / "models").glob("*.joblib"):
        if p.stat().st_size > 16 and "_2026" not in p.name:
            shutil.copy(p, test_out / "models" / p.name)
    tc = REPO / "outputs" / "reports" / "training_config.json"
    if tc.is_file():
        (test_out / "reports").mkdir(parents=True, exist_ok=True)
        shutil.copy(tc, test_out / "reports" / tc.name)
    env = dict(os.environ, PHISH_OUTPUTS_DIR=str(test_out), PHISH_DATA_DIR=str(WORK / "test_data"), COVERAGE_FILE=str(WORK / ".coverage"))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    runtime = time.perf_counter() - t0
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    tests = int(suite.get("tests"))
    failures = int(suite.get("failures"))
    errors = int(suite.get("errors"))
    skipped = int(suite.get("skipped"))
    not_passed = []
    for tc in suite.iter("testcase"):
        for ch in tc:
            if ch.tag in ("failure", "error", "skipped"):
                not_passed.append({"test": f"{tc.get('classname')}::{tc.get('name')}", "outcome": ch.tag, "message": (ch.get("message") or "")[:200]})
    covd = json.loads(cov.read_text())["totals"]
    write_result(
        "00_tests",
        {
            "command": "python -m pytest -q --cov=src (via metrics/scripts/00_tests.py)",
            "collected": tests,
            "passed": tests - failures - errors - skipped,
            "failed": failures,
            "errors": errors,
            "skipped": skipped,
            "pass_rate": (tests - failures - errors - skipped) / tests if tests else None,
            "line_coverage_percent_src": round(covd["percent_covered"], 1),
            "covered_lines": covd["covered_lines"],
            "num_statements": covd["num_statements"],
            "runtime_seconds": round(runtime, 1),
            "not_passed": not_passed,
            "pytest_exit_code": proc.returncode,
            "requires": "data/reference/*.csv (gitignored before this audit) and outputs/models/*.joblib + outputs/reports/training_config.json for the model smoke tests",
        },
    )


if __name__ == "__main__":
    main()
