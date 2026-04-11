"""Load Kaggle phishing+legitimate URL dataset (primary ML training source).

Supports:
1. Manual drop: place CSV (or ``*.csv``) under ``data/raw/kaggle/``.
2. Optional download via ``kagglehub`` when installed and credentials allow.

Only **URL** and **label** columns are kept for our pipeline (we compute Layer-1 features ourselves).
Extra columns from the Kaggle file are dropped to avoid accidental use of third-party engineered features.

Label verification:
- Detects candidate URL / label columns by name.
- Writes ``outputs/reports/kaggle_label_audit.json`` with class counts and samples.
- Maps Kaggle-style ``1=legitimate, 0=phishing`` → internal ``0=legit, 1=phish`` via :mod:`src.pipeline.label_policy`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.label_policy import INTERNAL_LEGIT, INTERNAL_PHISH, kaggle_status_to_internal
from src.pipeline.paths import ensure_layout, interim_dir, raw_dir, reports_dir

logger = logging.getLogger(__name__)

KAGGLE_SUBDIR = "kaggle"
SOURCE_DATASET = "kaggle_harisudhan411_phishing_and_legitimate_urls"

_URL_CANDIDATES = ("url", "link", "uri", "website", "webpage", "domain")
_LABEL_CANDIDATES = (
    "label",
    "class",
    "result",
    "status",
    "type",
    "classlabel",
    "target",
    "phishing",
)


def kaggle_raw_dir() -> Path:
    return raw_dir() / KAGGLE_SUBDIR


def _find_csv_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files = sorted(root.glob("*.csv")) + sorted(root.glob("**/*.csv"))
    # de-dupe same file
    seen = set()
    out: List[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _pick_url_column(df: pd.DataFrame) -> str:
    lower = {c.lower().strip(): c for c in df.columns}
    for key in _URL_CANDIDATES:
        if key in lower:
            return lower[key]
    # fallback: first column containing 'url' in name
    for c in df.columns:
        if "url" in c.lower():
            return c
    raise ValueError(f"Could not detect URL column. Columns: {list(df.columns)}")


def _pick_label_column(df: pd.DataFrame, url_col: str) -> str:
    lower = {c.lower().strip(): c for c in df.columns}
    for key in _LABEL_CANDIDATES:
        if key in lower and lower[key] != url_col:
            return lower[key]
    for c in df.columns:
        if c == url_col:
            continue
        cl = c.lower()
        if any(x in cl for x in ("label", "class", "phish", "legit", "status", "target")):
            return c
    raise ValueError(f"Could not detect label column. Columns: {list(df.columns)}")


def _normalize_binary_labels(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    uniq = sorted(s.dropna().unique().tolist())
    # allow 0/1 only
    bad = [x for x in uniq if x not in (0, 1)]
    if bad:
        raise ValueError(f"Non-binary label values after coercion: {bad[:10]}")
    return s


def verify_and_summarize_kaggle(
    df: pd.DataFrame, url_col: str, label_col: str, raw_label_name: str
) -> Dict[str, Any]:
    """Build audit dict; assert two classes present."""
    urls = df[url_col].astype(str)
    raw = _normalize_binary_labels(df[label_col])
    mask = raw.notna() & urls.str.strip().astype(bool)
    sub = df.loc[mask].copy()
    sub["_raw_lbl"] = raw[mask]

    counts_raw = sub["_raw_lbl"].value_counts().sort_index().to_dict()
    counts_raw = {int(k): int(v) for k, v in counts_raw.items()}

    internal = sub["_raw_lbl"].map(lambda x: kaggle_status_to_internal(int(x)))
    counts_internal = internal.value_counts().sort_index().to_dict()
    counts_internal = {int(k): int(v) for k, v in counts_internal.items()}

    if len(counts_internal) < 2:
        raise ValueError(
            f"Need both classes after cleaning; internal counts={counts_internal}. "
            "Check label column semantics or filtering."
        )

    # Small stratified samples for human spot-check (not semantic proof)
    samples: Dict[str, List[Dict[str, Any]]] = {"raw_0": [], "raw_1": []}
    for cls in (0, 1):
        part = sub[sub["_raw_lbl"] == cls].head(5)
        for _, r in part.iterrows():
            samples[f"raw_{cls}"].append(
                {
                    "url_preview": str(r[url_col])[:120],
                    "raw_label": int(r["_raw_lbl"]),
                    "internal_label": int(kaggle_status_to_internal(int(r["_raw_lbl"]))),
                }
            )

    audit: Dict[str, Any] = {
        "url_column": url_col,
        "raw_label_column": raw_label_name,
        "kaggle_label_semantics_documented": "1=legitimate, 0=phishing (Kaggle-style)",
        "internal_label_semantics": "0=legitimate, 1=phishing",
        "row_count_used": int(len(sub)),
        "raw_label_counts": counts_raw,
        "internal_label_counts": counts_internal,
        "stratified_samples": samples,
        "note": (
            "Verify raw_label_counts match the dataset documentation. "
            "If inverted, flip mapping in label_policy.kaggle_status_to_internal."
        ),
    }
    return audit


def load_kaggle_dataframe(csv_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Path]:
    """Load CSV from path or discover under data/raw/kaggle/."""
    ensure_layout()
    if csv_path is not None:
        path = csv_path
        if not path.is_file():
            raise FileNotFoundError(path)
    else:
        files = _find_csv_files(kaggle_raw_dir())
        if not files:
            raise FileNotFoundError(
                f"No CSV found under {kaggle_raw_dir()}. "
                "Add the Kaggle CSV there, or pass --csv path, or install kagglehub and use --download."
            )
        path = files[0]
        if len(files) > 1:
            logger.warning("Multiple CSVs in kaggle dir; using %s (first match)", path)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    return df, path


def try_download_kaggle_dataset() -> Optional[Path]:
    try:
        import kagglehub  # type: ignore
    except ImportError:
        logger.info("kagglehub not installed; skip automatic download.")
        return None
    try:
        p = kagglehub.dataset_download("harisudhan411/phishing-and-legitimate-urls")
        root = Path(p)
        csvs = sorted(root.rglob("*.csv"))
        if not csvs:
            logger.warning("kagglehub download produced no CSV under %s", root)
            return None
        dest_root = kaggle_raw_dir()
        dest_root.mkdir(parents=True, exist_ok=True)
        # copy first/largest csv
        csvs.sort(key=lambda x: x.stat().st_size, reverse=True)
        src = csvs[0]
        dest = dest_root / src.name
        dest.write_bytes(src.read_bytes())
        logger.info("Copied dataset CSV to %s", dest)
        return dest
    except Exception as e:
        logger.warning("kagglehub download failed: %s", e)
        return None


def ingest_kaggle(
    csv_path: Optional[Path] = None,
    *,
    download: bool = False,
) -> Path:
    ensure_layout()
    if download and csv_path is None:
        dl = try_download_kaggle_dataset()
        if dl is not None:
            csv_path = dl

    df, used_path = load_kaggle_dataframe(csv_path)
    url_col = _pick_url_column(df)
    label_col = _pick_label_column(df, url_col)

    audit = verify_and_summarize_kaggle(df, url_col, label_col, label_col)
    audit["source_csv"] = str(used_path.resolve())

    raw_lbl = _normalize_binary_labels(df[label_col])
    out = pd.DataFrame(
        {
            "url": df[url_col].astype(str).str.strip(),
            "label": raw_lbl.map(lambda x: kaggle_status_to_internal(int(x)) if pd.notna(x) else None),
            "source_file": used_path.name,
            "source_dataset": SOURCE_DATASET,
            "source_brand_hint": "",
            "action_category": "kaggle_primary",
            "kaggle_raw_status": raw_lbl.astype("Int64"),
        }
    )
    out = out.dropna(subset=["url", "label"])
    out["label"] = pd.to_numeric(out["label"], errors="coerce").astype(int)
    out = out[out["url"].str.len() > 0]

    audit["output_rows"] = int(len(out))
    audit["internal_label_counts_final"] = out["label"].value_counts().sort_index().to_dict()

    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "kaggle_label_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("Kaggle label audit -> outputs/reports/kaggle_label_audit.json")

    out_path = interim_dir() / "kaggle_normalized.csv"
    out.to_csv(out_path, index=False)
    logger.info(
        "Wrote %s rows (legit=%s phish=%s) -> %s",
        len(out),
        int((out["label"] == INTERNAL_LEGIT).sum()),
        int((out["label"] == INTERNAL_PHISH).sum()),
        out_path,
    )
    return out_path


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "kaggle_ingest.log")
    ap = argparse.ArgumentParser(description="Ingest Kaggle phishing+legitimate URLs (primary ML source).")
    ap.add_argument("--csv", type=Path, default=None, help="Explicit path to Kaggle CSV")
    ap.add_argument("--download", action="store_true", help="Try kagglehub download into data/raw/kaggle/")
    args = ap.parse_args()
    ingest_kaggle(args.csv, download=args.download)


if __name__ == "__main__":
    main()
