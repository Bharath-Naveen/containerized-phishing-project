# Dataset setup

## Primary training (Kaggle)

Dataset: **`harisudhan411/phishing-and-legitimate-urls`** (binary URL + label).

### Option A — Manual download

1. Download the CSV from Kaggle.
2. Place it under **`data/raw/kaggle/`** (any filename ending in `.csv`).
3. Verify labels in **`outputs/reports/kaggle_label_audit.json`** after first ingest (Kaggle convention: `1` = legitimate, `0` = phishing; internally mapped to `0` / `1` — see `src/pipeline/label_policy.py`).

### Option B — `kagglehub`

```bash
pip install kagglehub
python -m src.pipeline.kaggle_ingest --download
```

Requires Kaggle API credentials (`~/.kaggle/kaggle.json` or env vars — see `.env.example`).

### Train

From repo root with `PYTHONPATH` set (or use Docker):

```bash
python -m src.pipeline.run_kaggle_pipeline --limit 20000   # drop --limit for full set
```

Artifacts go to `data/processed/` and `outputs/` (gitignored).

---

## Challenge / brand lists (optional)

Sample lists live in **`archive/sample_data/challenge/`**. Copy into **`data/raw/`** when you want to run challenge ingest (see root `README.md`).

Do **not** commit copied files under `data/raw/`; they remain local.

---

## Internal label convention

| Meaning | `label` value |
|---------|----------------|
| Legitimate | `0` |
| Phishing | `1` |

Kaggle raw columns are converted during `kaggle_ingest`.
