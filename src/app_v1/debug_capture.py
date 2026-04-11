"""Run only the Playwright capture stage for a single URL (smoke / isolation).

Usage (PowerShell, from repo root):

  cd src; python -m app_v1.debug_capture https://example.com

Verbose step logging (before/after each capture sub-step):

  cd src; python -m app_v1.debug_capture https://example.com -v

Or run the file directly (adds ``src`` to ``sys.path``):

  python src/app_v1/debug_capture.py https://example.com -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python src/app_v1/debug_capture.py` by putting `src` on path.
_SRC_PARENT = Path(__file__).resolve().parents[1]
if str(_SRC_PARENT) not in sys.path:
    sys.path.insert(0, str(_SRC_PARENT))

from app_v1.capture import capture_url  # noqa: E402
from app_v1.config import PipelineConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture-stage smoke test for one URL.")
    parser.add_argument("url", help="Page URL to capture")
    parser.add_argument(
        "-n",
        "--namespace",
        default="suspicious",
        choices=("suspicious", "legit"),
        help="Artifact subdirectory under output_dir (default: suspicious)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging (includes capture step before/after lines)",
    )
    args = parser.parse_args()

    root_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=root_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Always show capture step before/after lines; -v raises root to DEBUG for everything else.
    logging.getLogger("app_v1.capture").setLevel(logging.INFO)

    cfg = PipelineConfig.from_env()
    out = cfg.ensure_output_dir()
    print(f"output_dir: {out}")
    print(f"namespace: {args.namespace}")
    print(f"url: {args.url}")
    if not args.verbose:
        print("(app_v1.capture steps: INFO; use -v for full DEBUG including asyncio/playwright.)")
    print("---")

    result = capture_url(args.url, cfg, namespace=args.namespace)
    j = result.as_json()

    print("CaptureResult fields:")
    for key in (
        "original_url",
        "final_url",
        "title",
        "screenshot_path",
        "fullpage_screenshot_path",
        "html_path",
        "capture_blocked",
        "capture_strategy",
        "capture_block_reason",
        "capture_block_evidence",
        "first_failed_capture_step",
        "error",
    ):
        print(f"  {key}: {j.get(key)!r}")

    vt = j.get("visible_text") or ""
    print(f"  visible_text: ({len(vt)} chars) {vt[:200]!r}{'...' if len(vt) > 200 else ''}")

    inter = j.get("interaction") or {}
    print("interaction:")
    for k, v in inter.items():
        print(f"  {k}: {v!r}")

    print("---")
    paths = [
        j.get("screenshot_path"),
        j.get("fullpage_screenshot_path"),
        j.get("html_path"),
        inter.get("post_submit_viewport_screenshot_path"),
    ]
    for p in paths:
        if not p:
            print(f"path (empty): {p!r}")
            continue
        exists = Path(p).exists()
        print(f"path: {p}  exists={exists}")

    if result.error:
        print(f"ERROR: {result.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
