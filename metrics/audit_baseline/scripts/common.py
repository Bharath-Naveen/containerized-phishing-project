"""Shared helpers for the metrics verification scripts.

All scripts write intermediate artifacts under metrics/_work/ (gitignored) so they never
overwrite the deployed models in outputs/models or the data in data/processed.
Final numbers go to metrics/results/*.json.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
REPO = Path(__file__).resolve().parents[3]
METRICS = REPO / "metrics" / "audit_baseline"
RESULTS = METRICS / "results"
WORK = REPO / "metrics" / "_work"
RAW_KAGGLE_CSV = REPO / "data" / "raw" / "kaggle" / "phishing_and_legitimate_urls.csv"


def isolate_env() -> None:
    """Point the pipeline at metrics/_work so nothing in outputs/ or data/processed is touched."""
    os.environ.setdefault("PHISH_DATA_DIR", str(WORK / "data"))
    os.environ.setdefault("PHISH_OUTPUTS_DIR", str(WORK / "outputs"))
    os.environ.setdefault("PHISH_LOGS_DIR", str(WORK / "logs"))
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    for d in ("data", "outputs", "logs"):
        (WORK / d).mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for p in (str(REPO / "src"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)


def seed_everything() -> None:
    random.seed(SEED)
    try:
        import numpy as np

        np.random.seed(SEED)
    except Exception:
        pass


def git_commit() -> str:
    try:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, env=env, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-c", "core.fileMode=false", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def code_trees() -> dict:
    """Git tree hashes of the code that produced the numbers (stable across README-only commits)."""
    out = {}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    for p in ("src", "tests", "data/evaluation", "requirements.txt"):
        try:
            out[p] = subprocess.run(
                ["git", "rev-parse", f"HEAD:{p}"], cwd=REPO, capture_output=True, text=True, env=env, check=True
            ).stdout.strip()
        except Exception:
            out[p] = "unknown"
    return out


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    from importlib import metadata

    pkgs = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib", "tldextract", "pytest", "pytest-cov", "playwright"):
        try:
            pkgs[name] = metadata.version(name)
        except Exception:
            pkgs[name] = "not installed"
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "ram_gb": mem_gb,
        "gpu": "none (CPU only)",
        "packages": pkgs,
    }


def write_result(name: str, payload: dict) -> Path:
    payload = dict(payload)
    payload.setdefault("_provenance", {})
    payload["_provenance"].update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "code_trees": code_trees(),
            "seed": SEED,
            "environment": environment(),
        }
    )
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return out
