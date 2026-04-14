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

### Train (Layer-1 pipeline)

The Kaggle dump is large (~800k+ rows deduplicated). **By default** the pipeline takes a **stratified sample of 50,000 rows** (class proportions preserved), writes it to `data/processed/cleaned_kaggle_sample_n*_rs*.csv`, and logs **`STRATIFIED_SAMPLE`** vs **`FULL_DATASET`**. Full metadata: `outputs/reports/kaggle_sample_manifest.json`.

From repo root with `PYTHONPATH` set (or use Docker):

```bash
# Default: stratified sample of 50k (fast iteration)
python -m src.pipeline.run_kaggle_pipeline

# Explicit sizes
python -m src.pipeline.run_kaggle_pipeline --sample-size 5000      # smoke
python -m src.pipeline.run_kaggle_pipeline --sample-size 200000    # larger eval
python -m src.pipeline.run_kaggle_pipeline --sample-frac 0.1       # 10% of deduped rows

# Full ~796k rows (many hours of Layer-1 enrich)
python -m src.pipeline.run_kaggle_pipeline --full

# Reproducibility
python -m src.pipeline.run_kaggle_pipeline --random-seed 42 --sample-size 50000

# Fresh enrich checkpoint for this sample (ignore stale interim checkpoint)
python -m src.pipeline.run_kaggle_pipeline --no-enrich-resume --sample-size 20000
```

Optional **`--limit`** still caps how many rows **enrich** processes after sampling (debug only).

**Suggested progression**

| Stage | Command idea | Purpose |
|-------|----------------|--------|
| Smoke | `--sample-size 2000`–`5000` | Pipeline wiring, minutes |
| Baseline | default (50k) or `--sample-size 50000` | Model comparison, tens of minutes |
| Larger eval | `--sample-size 200000` or `--sample-frac 0.25` | Better metrics, still bounded |
| Production-scale | `--full` | All deduplicated URLs; plan overnight / cluster |

**Layer-1 cost notes:** Enrichment is **URL parse + tldextract only** (no HTTP, no Playwright) unless you pass `--layer1-use-dns`. Checkpoints default to every **400** rows for Layer-1 to reduce disk I/O. `domain_hash_bucket` uses a **stable hash** (not Python’s `hash()`) so features are reproducible across processes.

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
