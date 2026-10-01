# Docker: reproducible capture (Linux + Playwright)

The image is based on [Playwright’s official Python image](https://playwright.dev/docs/docker) (Chromium + OS deps). `requirements.txt` pins `playwright==1.49.0` to match the `Dockerfile` base tag.

## Prerequisites

- Docker and Docker Compose v2 (`docker compose`)

## Build

```bash
docker compose build
```

Or:

```bash
docker build -t phishing-triage:app-v1 .
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `PHISH_OUTPUT_DIR` | Capture output root for triage UI (default in container: `/data/captures`). |
| `PHISH_PROJECT_ROOT` | Repo root inside container (`/app`). |
| `PHISH_DATA_DIR` | Pipeline data root (`/app/data`). |
| `PHISH_OUTPUTS_DIR` | Models/metrics (`/app/outputs`). |
| `PHISH_LOGS_DIR` | Logs (`/app/logs`). |
| `PHISH_NAV_TIMEOUT_MS`, `PHISH_WAIT_UNTIL`, etc. | Same as `PipelineConfig.from_env()` in `src/phishguard/app/runtime_config.py`. |

No API keys are needed to run the app. A `.env` file in the project root (Compose loads it automatically) can override the variables above; see `.env.example`.

## Run Streamlit dashboard (default)

The default UI is the **phishing analysis dashboard** (`src/phishguard/app/frontend.py`): Layer-1 ML + optional reinforcement.

## Run Streamlit frontend

```bash
docker compose up
```

Open [http://localhost:8501](http://localhost:8501).

Capture artifacts: `./captures` → `/data/captures`.

## Train the Layer-1 models

Run inside the `pipeline` service (the Kaggle CSV must be in `./data/raw/kaggle/`, see `docs/DATASET_SETUP.md`):

```bash
docker compose --profile pipeline run --rm pipeline python -m phishguard train                 # 50K stratified sample
docker compose --profile pipeline run --rm pipeline python -m phishguard train --full          # all ~796K rows
```

Score one URL from the command line:

```bash
docker compose run --rm triage python -m phishguard analyze --url "https://example.com"
```

## Run `debug_capture.py` (single URL)

From the project root:

```bash
docker compose run --rm triage python -m phishguard.app.debug_capture "https://example.com"
```

Verbose logs:

```bash
docker compose run --rm triage python -m phishguard.app.debug_capture "https://example.com" -v
```

## Volumes (host → container)

| Host | Container | Use |
|------|-----------|-----|
| `./captures` | `/data/captures` | Screenshots, HTML (`PHISH_OUTPUT_DIR`) |
| `./data` | `/app/data` | Raw/interim/processed CSVs for the ML pipeline |
| `./outputs` | `/app/outputs` | Models, metrics, figures |
| `./logs` | `/app/logs` | Pipeline logs |

Create host dirs if missing:

```bash
mkdir -p captures data outputs logs
```

## Headed / stealth capture in Docker

Strategy A may use `headless=False`. In a headless container there is no real display; capture falls back to HTTP or the next strategy. For headed testing, use Xvfb or a local Linux desktop—see Playwright docs.

## One-off shell

```bash
docker compose run --rm triage bash
```

Inside the container, `PYTHONPATH` is `/app:/app/src`.
