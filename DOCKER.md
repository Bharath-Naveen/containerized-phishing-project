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
| `OPENAI_API_KEY` | Optional; enables AI brand/task stage (Streamlit / `app_v1`) and optional LLM helpers. |
| `PHISH_OUTPUT_DIR` | Capture output root for triage UI (default in container: `/data/captures`). |
| `PHISH_PROJECT_ROOT` | Repo root inside container (`/app`). |
| `PHISH_DATA_DIR` | Pipeline data root (`/app/data`). |
| `PHISH_OUTPUTS_DIR` | Models/metrics (`/app/outputs`). |
| `PHISH_LOGS_DIR` | Logs (`/app/logs`). |
| `PHISH_NAV_TIMEOUT_MS`, `PHISH_WAIT_UNTIL`, etc. | Same as `PipelineConfig.from_env()` in `src/app_v1/config.py`. |

Create a `.env` file in the project root (Compose loads it automatically):

```env
OPENAI_API_KEY=sk-...
```

## Run Streamlit dashboard (default)

The default UI is the **phishing analysis dashboard** (`src/app_v1/frontend.py`): Layer-1 ML + optional reinforcement. Legacy screenshot-comparison UI: `archive/legacy/frontend_screenshot_legacy.py`.

## Run Streamlit frontend

```bash
docker compose up
```

Open [http://localhost:8501](http://localhost:8501).

Capture artifacts: `./captures` → `/data/captures`.

## Batch ML pipeline (scraping + DNS + training)

**Run these inside the `pipeline` service** so HTTP fetches and Playwright never execute on the host:

```bash
docker compose --profile pipeline run --rm pipeline python -m src.pipeline.run_all --limit 400
```

Individual stages:

```bash
docker compose --profile pipeline run --rm pipeline python -m src.pipeline.ingest
docker compose --profile pipeline run --rm pipeline python -m src.pipeline.clean
docker compose --profile pipeline run --rm pipeline python -m src.pipeline.enrich --limit 200
docker compose --profile pipeline run --rm pipeline python -m src.pipeline.split
docker compose --profile pipeline run --rm pipeline python -m src.pipeline.train
```

Passive Playwright behavior probe (optional, slower):

```bash
docker compose --profile pipeline run --rm pipeline python -m src.pipeline.enrich --playwright --limit 50
```

## Run `debug_capture.py` (single URL)

From the project root:

```bash
docker compose run --rm triage python -m app_v1.debug_capture "https://example.com"
```

Verbose logs:

```bash
docker compose run --rm triage python -m app_v1.debug_capture "https://example.com" -v
```

## Run orchestrator CLI

Write JSONL under the mounted data folder:

```bash
docker compose run --rm triage python -m app_v1.orchestrator --url "https://example.com" --out /app/data/triage_rows_v1.jsonl
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
