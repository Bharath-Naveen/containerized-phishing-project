# Phishing Detection Platform (Team Share)

This repository provides a layered phishing detection system for URL triage and page-level reinforcement:

- Layer 1: fast ML triage on URL/host features.
- Layer 2: optional live capture + HTML/DOM + host/path reasoning.
- Layer 3: bounded AI adjudication over compact evidence only.
- Layer 4: final explainable verdict with guardrails and evidence gaps.

The project is optimized for team collaboration: active code is under `src/app_v1` and `src/pipeline`, deprecated screenshot-first workflows are preserved under `archive/legacy` and clearly labeled.

## Why Screenshot Comparison Was Deprecated

The original screenshot-comparison UX was useful for demos, but weak for production triage because it:

- required brittle visual matching/reference maintenance,
- struggled with modern dynamic layouts and localization,
- offered limited explainability versus structured signals.

The current architecture favors structured, auditable evidence (DOM/form/link/host/path) and bounded AI support. Legacy screenshot code remains available in `archive/legacy` for reference.

## Current Layered Architecture

See full details in `docs/ARCHITECTURE.md`.

1. `src/pipeline`:
   - data ingestion, cleaning, feature extraction, splits, training/evaluation, audits.
2. `src/app_v1`:
   - runtime URL analysis, capture, DOM anomaly extraction, host/path reasoning, AI adjudication, Streamlit UI.
3. final output:
   - explainable JSON from `src/app_v1/analyze_dashboard.py`.

## Quick Start

### Local

```powershell
$env:PYTHONPATH = "$PWD"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pytest
streamlit run src/app_v1/frontend.py
```

### Docker (recommended for untrusted URLs)

```bash
docker compose build
docker compose up
# then open http://localhost:8501
```

More container details: `DOCKER.md`.

## Training Workflow

Primary dataset setup: `docs/DATASET_SETUP.md`.

Typical run:

```bash
python -m src.pipeline.run_kaggle_pipeline
```

Useful variants:

```bash
python -m src.pipeline.run_kaggle_pipeline --sample-size 5000
python -m src.pipeline.run_kaggle_pipeline --full
```

Pipeline overview: `docs/PIPELINE_OVERVIEW.md`.

## Running the Frontend

Main UI:

```bash
streamlit run src/app_v1/frontend.py
```

CLI JSON mode:

```bash
python -m src.app_v1.analyze_dashboard --url "https://example.com"
```

Frontend guide: `docs/FRONTEND_GUIDE.md`.

## Dashboard Sections (What They Mean)

- **Verdict**: final label/confidence/combined score with reasons.
- **Layer-1 ML**: URL/host classifier output and top linear signals.
- **Reinforcement**: live capture status + org-style risk.
- **HTML Structure Signals**: compact DOM structure summary.
- **HTML / DOM Anomaly Review**: mismatch/interstitial/harvester evidence.
- **Host / Path reasoning**: host legitimacy and path-conformity assessment.
- **AI Adjudication**: bounded post-check (if enabled and eligible).
- **Evidence gaps**: known blind spots from failed/partial capture.

## Strengths

- Explainable multi-layer evidence (not ML-only).
- Local-first parsing (raw full HTML is not sent to AI).
- Strong phishing guardrails (credential capture/deceptive host patterns).
- Bounded AI influence with auditable adjustments.
- Container-friendly execution for safer URL analysis.

## Known Limitations

- Layer-1 may still over-score some legitimate long-tail domains.
- Live capture quality depends on network/captcha/bot defenses.
- Heuristic layers can produce edge-case noise on unusual public pages.
- AI adjudication is optional and API-dependent.

## Repository Structure

```text
.
├── src/
│   ├── app_v1/                 # Live app runtime / dashboard / adjudication
│   └── pipeline/               # Data + training + audits
├── tests/                      # pytest coverage
├── docs/                       # Team docs
├── scripts/                    # Helper scripts
├── archive/
│   ├── legacy/                 # Deprecated screenshot-first code (kept, not used)
│   └── sample_data/challenge/  # Optional sample URL lists
├── data/                       # Local datasets (gitignored except placeholders)
├── outputs/                    # Models/reports/artifacts (gitignored)
├── captures/                   # Runtime captures (gitignored)
└── logs/                       # Logs (gitignored)
```

## Documentation Index

- `docs/ARCHITECTURE.md`
- `docs/FRONTEND_GUIDE.md`
- `docs/PIPELINE_OVERVIEW.md`
- `docs/CHANGELOG_PROJECT_EVOLUTION.md`
- `docs/TESTING.md`
- `docs/DATASET_SETUP.md`

## Security and Sharing Notes

- Never commit `.env`, API keys, raw datasets, captures, or outputs.
- Review `.gitignore` before pushing.
- Legacy code in `archive/legacy` is reference-only and deprecated.
