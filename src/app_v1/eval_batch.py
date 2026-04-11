"""Batch-run the phishing triage pipeline over many URLs; write JSONL + CSV summary.

CLI::

    python -m src.app_v1.eval_batch \\
        --input data/raw/phishing_dataset.csv \\
        --url-column url \\
        --out-jsonl data/eval_results.jsonl \\
        --out-csv data/eval_summary.csv \\
        --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .orchestrator import run_pipeline
from .schemas import utc_now_iso

SUMMARY_COLUMNS: List[str] = [
    "input_url",
    "final_url",
    "brand_guess",
    "task_guess",
    "trusted_reference_found",
    "verdict",
    "confidence",
    "behavior_gap_score",
    "post_submit_left_trusted_domain",
    "trusted_oauth_redirect",
    "error",
]


def _error_row(input_url: str, message: str) -> Dict[str, Any]:
    """Minimal JSON-serializable row when the pipeline cannot complete."""
    return {
        "timestamp_utc": utc_now_iso(),
        "input_url": input_url,
        "capture": {},
        "ai_brand_task": {},
        "legit_lookup": {},
        "legit_reference_capture": None,
        "features": {},
        "comparison": {},
        "verdict": {"verdict": "", "confidence": "", "reasons": []},
        "error": message,
    }


def load_urls(input_path: Path, url_column: str) -> List[str]:
    """Load non-empty URL strings from CSV (first column match) or JSONL."""
    suffix = input_path.suffix.lower()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(input_path)
        if url_column not in df.columns:
            lower_map = {str(c).lower(): c for c in df.columns}
            key = url_column.lower()
            if key in lower_map:
                col = lower_map[key]
            else:
                raise ValueError(
                    f"URL column {url_column!r} not in CSV. Columns: {list(df.columns)}"
                )
        else:
            col = url_column
        series = df[col].dropna().astype(str).str.strip()
        return [u for u in series.tolist() if u]

    if suffix == ".jsonl":
        urls: List[str] = []
        with input_path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL line {line_no}: invalid JSON: {exc}") from exc
                if url_column not in obj:
                    raise ValueError(
                        f"JSONL line {line_no}: missing key {url_column!r}. Keys: {list(obj.keys())}"
                    )
                u = str(obj[url_column]).strip()
                if u:
                    urls.append(u)
        return urls

    raise ValueError(f"Unsupported input extension {suffix!r}; use .csv or .jsonl")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_summary_row(full_row: Dict[str, Any]) -> Dict[str, str]:
    """Flatten nested pipeline output for evaluation CSV."""
    cap = full_row.get("capture") or {}
    ai = full_row.get("ai_brand_task") or {}
    comp = full_row.get("comparison") or {}
    ver = full_row.get("verdict") or {}
    return {
        "input_url": _csv_cell(full_row.get("input_url")),
        "final_url": _csv_cell(cap.get("final_url")),
        "brand_guess": _csv_cell(ai.get("brand_guess")),
        "task_guess": _csv_cell(ai.get("task_guess")),
        "trusted_reference_found": _csv_cell(comp.get("trusted_reference_found")),
        "verdict": _csv_cell(ver.get("verdict")),
        "confidence": _csv_cell(ver.get("confidence")),
        "behavior_gap_score": _csv_cell(comp.get("behavior_gap_score")),
        "post_submit_left_trusted_domain": _csv_cell(comp.get("post_submit_left_trusted_domain")),
        "trusted_oauth_redirect": _csv_cell(comp.get("trusted_oauth_redirect")),
        "error": _csv_cell(full_row.get("error")),
    }


def run_batch(
    input_path: Path,
    url_column: str,
    out_jsonl: Path,
    out_csv: Path,
    limit: Optional[int] = None,
) -> None:
    urls = load_urls(input_path, url_column)
    if limit is not None:
        urls = urls[: max(0, limit)]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, str]] = []

    with out_jsonl.open("w", encoding="utf-8", newline="\n") as jf:
        for url in urls:
            try:
                row = run_pipeline(url)
            except Exception as exc:  # noqa: BLE001
                row = _error_row(url, f"batch_fatal: {type(exc).__name__}: {exc}")
            summaries.append(build_summary_row(row))
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    with out_csv.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Processed {len(urls)} URL(s).", file=sys.stderr)
    print(f"JSONL: {out_jsonl}", file=sys.stderr)
    print(f"CSV:   {out_csv}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run phishing triage pipeline over many URLs (CSV or JSONL).",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .csv or .jsonl path")
    parser.add_argument(
        "--url-column",
        default="url",
        help="Column name (CSV) or object key (JSONL) for the URL",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/eval_results.jsonl"),
        help="One full pipeline JSON object per line",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/eval_summary.csv"),
        help="Compact per-URL summary for spreadsheets",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N URLs (default: all)",
    )
    args = parser.parse_args(argv)
    run_batch(
        input_path=args.input,
        url_column=args.url_column,
        out_jsonl=args.out_jsonl,
        out_csv=args.out_csv,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
