# Architecture

This document describes the **active architecture** used by the team and what is intentionally deprecated.

## High-Level Layers

1. **Layer 1: URL/host ML triage** (`src/pipeline`, runtime in `src/app_v1/ml_layer1.py`)
   - fast score from lexical/hosting/brand-structure features.
   - **Optional multi-model agreement:** `outputs/models/layer1_primary.joblib` remains the authoritative ML probability; additional `logistic_regression`, `random_forest`, `xgboost`, and `lightgbm` artifacts (when present) are scored as witness models. Consensus or disagreement is passed into the deterministic EAL as supporting evidence only, not as a replacement verdict.
2. **Layer 2: Reinforcement** (`src/app_v1/capture.py`, `org_style_signals.py`)
   - optional live fetch/capture + org-style checks.
3. **Layer 3: HTML/DOM + host/path reasoning**
   - `html_structure_signals.py`
   - `html_dom_anomaly_signals.py`
   - `host_path_reasoning.py`
4. **Layer 4: Deterministic Evidence Adjudication Layer (EAL)**
   - evidence taxonomy + inspectable scoring in `analyze_dashboard.py`.
5. **Final decision policy**
   - EAL verdict + bounded legitimacy rescue + hard no-phishing-evidence guard.

## Active Runtime Flow

`src/app_v1/analyze_dashboard.py`:

1. Run Layer-1 ML probability.
2. Optional reinforcement capture.
3. Extract HTML structure + DOM anomaly signals.
4. Compute host/path reasoning.
5. Blend ML + reinforcement with bounded discounts/guardrails.
6. Apply deterministic EAL scoring and hard-blocker policy.
7. Apply hard no-phishing-evidence override when all red flags are absent.
8. Emit explainable JSON payload for frontend and audits.

## Training/Eval Architecture

Primary path (`src/pipeline/run_kaggle_pipeline.py`):

`kaggle_ingest -> clean -> enrich --layer1-only -> split_leak_safe -> train --layer1-only`

Supporting utilities:

- leakage report: `src/pipeline/leakage_report.py`
- audits: `src/pipeline/fp_audit.py`, `src/pipeline/phish_audit.py`

## Deprecated Components

Deprecated screenshot-first workflow is retained under `archive/legacy`:

- `archive/legacy/frontend_screenshot_legacy.py`
- `archive/legacy/scripts/*`

These are **reference-only** and not part of the recommended runtime or training loop.

## Data Boundaries and Security

- Raw datasets live under `data/raw` (gitignored except placeholders).
- Intermediate/processed data live under `data/interim`, `data/processed` (gitignored except placeholders).
- Captures, outputs, logs are gitignored.
- Raw full HTML is not sent to AI; only compact summaries are used.

## Why This Architecture

- Keeps model speed (Layer-1) while adding explainability and correction layers.
- Separates train-time code (`src/pipeline`) from runtime app (`src/app_v1`).
- Preserves old code in archive without deleting team history.
