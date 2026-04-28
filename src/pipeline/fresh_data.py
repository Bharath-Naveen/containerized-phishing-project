"""Fresh data utilities for optional phishing/legitimate augmentation."""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests
import tldextract

logger = logging.getLogger(__name__)

PHISH_STATUS = 0
LEGIT_STATUS = 1


def get_registered_domain(url: str) -> str:
    ext = tldextract.extract(str(url or ""))
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return (ext.domain or "").lower()


def label_sanity_check(df: pd.DataFrame, *, status_col: str = "status") -> pd.DataFrame:
    out = df.copy()
    out[status_col] = pd.to_numeric(out.get(status_col), errors="coerce")
    out = out[out[status_col].isin([PHISH_STATUS, LEGIT_STATUS])].copy()
    out[status_col] = out[status_col].astype(int)
    return out


def deduplicate_urls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["url"] = out.get("url", "").astype(str).str.strip()
    out = out[out["url"].str.len() > 0].copy()
    out = out.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    return out


def cap_per_domain(df: pd.DataFrame, *, max_per_domain: int) -> pd.DataFrame:
    if max_per_domain <= 0:
        return df.iloc[0:0].copy()
    out = df.copy()
    if "registered_domain" not in out.columns:
        out["registered_domain"] = out["url"].map(get_registered_domain)
    out["registered_domain"] = out["registered_domain"].fillna("").astype(str).str.lower()
    capped = (
        out.groupby("registered_domain", dropna=False, group_keys=False)
        .head(max_per_domain)
        .reset_index(drop=True)
    )
    return capped


def collect_phishstats(
    *,
    pages: int = 8,
    timeout_s: int = 20,
    max_rows: Optional[int] = None,
    api_base_url: str = "https://api.phishstats.info/api/phishing",
    max_failed_pages: int = 3,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Collect recent phishing URLs from phishstats.info API.

    Returns an empty frame on network/API failures.
    """
    rows: list[dict] = []
    fetch_errors = 0
    consecutive_failed_pages = 0
    for p in range(1, max(1, pages) + 1):
        # PhishStats API commonly uses 0-indexed page numbers.
        page_idx = p - 1
        url = f"{api_base_url}?_sort=-id&_p={page_idx}&_size=100"
        try:
            resp = requests.get(url, timeout=timeout_s)
            if resp.status_code != 200:
                logger.warning("phishstats page=%s status=%s", p, resp.status_code)
                fetch_errors += 1
                consecutive_failed_pages += 1
                if consecutive_failed_pages >= max(1, int(max_failed_pages)):
                    logger.warning("phishstats stopping early after %s failed pages", consecutive_failed_pages)
                    break
                continue
            payload = resp.json()
            if not isinstance(payload, list):
                logger.warning("phishstats page=%s unexpected payload type", p)
                fetch_errors += 1
                consecutive_failed_pages += 1
                if consecutive_failed_pages >= max(1, int(max_failed_pages)):
                    logger.warning("phishstats stopping early after %s failed pages", consecutive_failed_pages)
                    break
                continue
            if not payload:
                break
            consecutive_failed_pages = 0
            for it in payload:
                u = str((it or {}).get("url") or "").strip()
                if not u:
                    continue
                rows.append(
                    {
                        "url": u,
                        "status": PHISH_STATUS,
                        "source": "phishstats",
                        "collection_date": datetime.now(UTC).date().isoformat(),
                        "registered_domain": get_registered_domain(u),
                    }
                )
                if max_rows is not None and len(rows) >= max_rows:
                    break
            if max_rows is not None and len(rows) >= max_rows:
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("phishstats fetch failed page=%s err=%s", p, exc)
            fetch_errors += 1
            consecutive_failed_pages += 1
            if consecutive_failed_pages >= max(1, int(max_failed_pages)):
                logger.warning("phishstats stopping early after %s failed pages", consecutive_failed_pages)
                break
            continue
    out_df = pd.DataFrame(rows, columns=["url", "status", "source", "collection_date", "registered_domain"])
    meta = {
        "phishstats_rows_collected": int(len(out_df)),
        "phishstats_fetch_errors": int(fetch_errors),
        "phishstats_pages_attempted": int(max(1, pages)),
    }
    if return_meta:
        return out_df, meta
    return out_df


def collect_tranco(
    *,
    n: int = 2000,
    csv_url: str = "https://tranco-list.eu/top-1m.csv.zip",
    timeout_s: int = 20,
    local_csv_path: Optional[Path] = None,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Collect likely-legitimate domains from Tranco list.

    Returns empty frame on failure.
    """
    tranco_error: Optional[str] = None
    tranco_download_failed = False
    try:
        if local_csv_path is not None and Path(local_csv_path).is_file():
            df = pd.read_csv(local_csv_path, header=None, names=["rank", "domain"], nrows=max(1, n))
        else:
            resp = requests.get(csv_url, timeout=timeout_s)
            if resp.status_code != 200:
                tranco_download_failed = True
                tranco_error = f"http_status_{resp.status_code}"
                logger.warning("tranco status=%s", resp.status_code)
                empty = pd.DataFrame(columns=["url", "status", "source", "collection_date", "registered_domain"])
                meta = {
                    "tranco_rows_collected": 0,
                    "tranco_download_failed": tranco_download_failed,
                    "tranco_error": tranco_error,
                }
                if return_meta:
                    return empty, meta
                return empty
            content_type = str(resp.headers.get("content-type", "")).lower()
            if "zip" in content_type or csv_url.lower().endswith(".zip"):
                import zipfile

                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    names = [x for x in zf.namelist() if x.lower().endswith(".csv")]
                    if not names:
                        raise ValueError("tranco_zip_missing_csv")
                    with zf.open(names[0]) as f:
                        df = pd.read_csv(f, header=None, names=["rank", "domain"], nrows=max(1, n))
            else:
                df = pd.read_csv(io.StringIO(resp.text), header=None, names=["rank", "domain"], nrows=max(1, n))
        rows: list[dict] = []
        for d in df["domain"].astype(str).tolist():
            d = d.strip().lower()
            if not d:
                continue
            url = f"https://{d}/"
            rows.append(
                {
                    "url": url,
                    "status": LEGIT_STATUS,
                    "source": "tranco",
                    "collection_date": datetime.now(UTC).date().isoformat(),
                    "registered_domain": get_registered_domain(url),
                }
            )
        out_df = pd.DataFrame(rows, columns=["url", "status", "source", "collection_date", "registered_domain"])
        meta = {
            "tranco_rows_collected": int(len(out_df)),
            "tranco_download_failed": tranco_download_failed,
            "tranco_error": tranco_error,
        }
        if return_meta:
            return out_df, meta
        return out_df
    except Exception as exc:  # noqa: BLE001
        logger.warning("tranco fetch failed err=%s", exc)
        empty = pd.DataFrame(columns=["url", "status", "source", "collection_date", "registered_domain"])
        meta = {
            "tranco_rows_collected": 0,
            "tranco_download_failed": True,
            "tranco_error": str(exc),
        }
        if return_meta:
            return empty, meta
        return empty


def ensure_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "registered_domain" not in out.columns:
        out["registered_domain"] = out["url"].map(get_registered_domain)
    if "status" not in out.columns:
        out["status"] = pd.NA
    return label_sanity_check(out)


def count_by(df: pd.DataFrame, cols: Iterable[str]) -> list[dict]:
    if df.empty:
        return []
    c = [x for x in cols if x in df.columns]
    if not c:
        return []
    return df.groupby(c).size().rename("n").reset_index().to_dict(orient="records")
