"""Optional model download helpers for demo/runtime enrichments."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
_DEFAULT_RELATIVE_MODEL_PATH = Path("models") / "lid.176.ftz"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_fasttext_model_path() -> Path:
    env_path = os.getenv("PHISH_FASTTEXT_LID_MODEL", "").strip()
    if env_path:
        return Path(env_path)
    return _repo_root() / _DEFAULT_RELATIVE_MODEL_PATH


def download_fasttext_model(*, overwrite: bool = False) -> Tuple[bool, str, Optional[Path]]:
    dest = _repo_root() / _DEFAULT_RELATIVE_MODEL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not overwrite:
        return True, f"already_exists:{dest}", dest
    try:
        with urllib.request.urlopen(_FASTTEXT_URL, timeout=60) as resp:
            payload = resp.read()
        if not payload:
            return False, "download_empty_payload", None
        dest.write_bytes(payload)
        return True, f"downloaded:{dest}", dest
    except urllib.error.URLError as exc:
        return False, f"download_failed:{type(exc).__name__}:{exc}", None
    except OSError as exc:
        return False, f"write_failed:{type(exc).__name__}:{exc}", None


def ensure_fasttext_model(*, trigger_download: bool = False) -> Tuple[bool, Optional[Path], Optional[str]]:
    """Resolve fastText model path; optionally download if missing.

    Returns (available, model_path, error_message).
    """
    model_path = resolve_fasttext_model_path()
    if model_path.is_file():
        return True, model_path, None
    if not trigger_download:
        return False, None, f"fasttext_model_not_found:{model_path}"
    ok, msg, p = download_fasttext_model(overwrite=False)
    if ok and p is not None and p.is_file():
        return True, p, None
    return False, None, msg


def main() -> None:
    ap = argparse.ArgumentParser(description="Optional model download helper.")
    ap.add_argument("--fasttext", action="store_true", help="Download fastText lid.176.ftz into ./models/")
    args = ap.parse_args()
    if not args.fasttext:
        print("No model action selected. Use --fasttext.")
        return
    ok, msg, _ = download_fasttext_model(overwrite=False)
    if ok:
        print(f"fasttext: {msg}")
    else:
        print(f"fasttext: {msg}")


if __name__ == "__main__":
    main()
