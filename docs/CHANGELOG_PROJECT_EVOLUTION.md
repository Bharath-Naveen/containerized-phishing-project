# Project Evolution Changelog

This is a high-level technical evolution log for teammates onboarding to the current codebase.

## Phase 1: Initial Screenshot-Oriented Prototype

- capture + compare approach with screenshot/reference matching
- early one-off scripts and manual workflows
- limited explainability and brittle visual dependence

Current status: retained under `archive/legacy`, not the recommended production path.

## Phase 2: Layer-1 ML Pipeline Stabilization

- formalized ingestion/clean/enrich/split/train pipeline in `src/pipeline`
- standardized label mapping and leak-safe splitting
- added repeatable reports and model artifacts

## Phase 3: App v1 Runtime and Dashboard

- introduced `src/app_v1/analyze_dashboard.py` as JSON runtime entrypoint
- introduced Streamlit UI `src/app_v1/frontend.py`
- added evidence gaps and per-layer explanations

## Phase 4: Explainable Reinforcement and AI Guardrails

- org-style reinforcement scoring
- bounded AI adjudication (non-primary, support-only)
- pre/post AI audit traces

## Phase 5: HTML/DOM + Host/Path Reasoning

- HTML structure extraction
- richer DOM anomaly reasoning (link/form/resource mismatch, interstitial, harvester patterns)
- page-family-aware dampening on content/reference pages
- host identity + path conformity scoring
- same-ecosystem domain grouping for link/form target interpretation

## Phase 6: False-Positive Mitigation and Safety Overrides

- bounded legitimacy rescue blending
- high-legitimacy content rescue path
- hard no-phishing-evidence guard override
- stronger separation between phishing red flags and benign public-content behavior

## Current Recommended Path

1. Train/evaluate with `src/pipeline`.
2. Run runtime analysis with `src/app_v1`.
3. Keep deprecated visual comparison in `archive/legacy` only for reference.
