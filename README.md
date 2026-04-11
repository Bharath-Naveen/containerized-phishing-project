# Phishing URL analysis (Layer-1 ML + reinforcement)

Container-friendly project: **fast URL/host ML triage** (primary), optional **live fetch + org-style signals** (secondary), and a **Streamlit dashboard** for verdicts and evidence gaps. Screenshot-vs-reference comparison is **deprecated**; legacy UI lives under `archive/legacy/`.

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, data flow, what is deprecated |
| [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md) | Kaggle CSV placement / download |
| [docs/TESTING.md](docs/TESTING.md) | Test strategy & manual checks |
| [DOCKER.md](DOCKER.md) | Build & run in Docker |

## Quick start

```powershell
# Repo root
$env:PYTHONPATH = "$PWD"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

```bash
export PYTHONPATH="$PWD"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Copy **`.env.example` → `.env`** only for local secrets (never commit `.env`). See [`.env.example`](.env.example).

## Repository layout

```
├── src/
│   ├── pipeline/          # Batch ML: ingest, clean, enrich, split, train
│   └── app_v1/            # Dashboard, capture, reinforcement helpers
├── tests/                 # pytest
├── scripts/               # Optional helpers (challenge sample copy)
├── docs/                  # Architecture, dataset, testing
├── archive/
│   ├── legacy/            # Screenshot-first Streamlit + old scripts
│   └── sample_data/challenge/   # Sample legit/phish .txt lists (copy to data/raw/)
├── data/                  # Gitignored content except READMEs / .gitkeep
├── outputs/               # Models, metrics, reports (gitignored)
└── logs/                  # gitignored
```

## 1. Kaggle dataset (primary training)

1. Place the CSV from **`harisudhan411/phishing-and-legitimate-urls`** under **`data/raw/kaggle/`**, **or** run `python -m src.pipeline.kaggle_ingest --download` (requires [Kaggle credentials](https://www.kaggle.com/docs/api)).

2. Run the full Layer-1 pipeline:

```bash
python -m src.pipeline.run_kaggle_pipeline --limit 20000   # omit --limit for full data
```

Outputs: `data/processed/*`, `outputs/models/layer1_primary.joblib`, `outputs/metrics/metrics.json`, reports under `outputs/reports/`.

Details: **[docs/DATASET_SETUP.md](docs/DATASET_SETUP.md)**.

## 2. Train the ML model (summary)

- **Primary path:** `run_kaggle_pipeline` (above) — includes train with **`--layer1-only`** internally.
- **Manual steps:** `kaggle_ingest` → `clean` → `enrich --layer1-only` → `split_leak_safe` → `train --layer1-only` (see `src/pipeline/run_kaggle_pipeline.py`).

## 3. Run the app

**Docker (recommended for untrusted URLs):**

```bash
docker compose build
docker compose up
# http://localhost:8501
```

**Local:**

```bash
streamlit run src/app_v1/frontend.py
```

CLI JSON (no UI):

```bash
cd src && python -m app_v1.analyze_dashboard --url "https://example.com"
```

## 4. Challenge-set evaluation (optional brand lists)

Sample URL lists are in **`archive/sample_data/challenge/`**. Copy into **`data/raw/`**, then:

```bash
python -m src.pipeline.ingest_challenge
```

Produces **`data/interim/challenge_normalized.csv`**. Continue with `clean` / `enrich` on that file if you want features (adjust `--input` / `--output` paths), or use the rows for custom eval. Helper:

```powershell
.\scripts\copy_challenge_samples.ps1
```

## 5. Tests

```bash
pytest
```

See **[docs/TESTING.md](docs/TESTING.md)** for pipeline, ML-only, reinforcement, and E2E expectations.

## Security

Analyze **untrusted URLs** inside **Docker**. No real credentials are submitted; captures are passive.

## License / data

Do not commit Kaggle CSVs, processed tables, API keys, or `kaggle.json`. See **`.gitignore`**.
