"""Optional extra legitimate training data: homepages of the Tranco top sites (rebuild Phase 3).

Why: the Kaggle legitimate rows do not look like many modern official sites, so the URL model
flags about a third of real official brand URLs. Tranco (https://tranco-list.eu) is a research
ranking of popular registered domains that is hard to manipulate; its top domains are a reasonable
source of "definitely a real site" homepages.

Reproducibility: the list is pinned by its Tranco list ID and the downloaded file's sha256 is
recorded. Rows are homepages only (https://<domain>/), so they teach "popular real domain", not
paths. Evaluation domains are still removed afterwards by drop_evaluation_rows().
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from phishguard.data.labels import INTERNAL_LEGIT
from phishguard.paths import raw_dir
from phishguard.urls.normalize import canonical_url

logger = logging.getLogger(__name__)

DEFAULT_LIST_ID = "K9QPW"  # Tranco list generated 2026-09-01 (30-day Dowdall combination, 5 providers)


def tranco_csv_path(list_id: str = DEFAULT_LIST_ID) -> Path:
    return raw_dir() / "tranco" / f"tranco_{list_id}.csv"


def fetch_tranco(list_id: str = DEFAULT_LIST_ID) -> Path:
    """Download the pinned list once (a zip or csv with 'rank,domain' rows) and cache it under data/raw/tranco/."""
    out = tranco_csv_path(list_id)
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://tranco-list.eu/download/{list_id}/1000000"
    logger.info("Downloading Tranco list %s from %s", list_id, url)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed https host)
        body = resp.read()
    if body[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            body = z.read(z.namelist()[0])
    out.write_bytes(body)
    return out


def tranco_rows(top_n: int, base_columns, list_id: str = DEFAULT_LIST_ID) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = fetch_tranco(list_id)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    domains = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1].strip():
                domains.append(row[1].strip().lower())
            if len(domains) >= top_n:
                break
    recs = []
    for d in domains:
        u = f"https://{d}/"
        c, inv, err = canonical_url(u)
        rec = {col: "" for col in base_columns}
        rec.update({"url": u, "label": str(INTERNAL_LEGIT), "canonical_url": c, "invalid_url": inv, "parse_error": err})
        if "source_dataset" in rec:
            rec["source_dataset"] = f"tranco_{list_id}"
        if "source_file" in rec:
            rec["source_file"] = path.name
        if "action_category" in rec:
            rec["action_category"] = "tranco_top_site_homepage"
        if "kaggle_raw_status" in rec:
            rec["kaggle_raw_status"] = "1"
        recs.append(rec)
    df = pd.DataFrame(recs, columns=list(base_columns))
    stats = {"list_id": list_id, "list_url": f"https://tranco-list.eu/list/{list_id}/1000000", "file_sha256": sha,
             "top_n_requested": top_n, "rows_built": int(len(df))}
    return df, stats
