# Pipeline Overview (`src/pipeline`)

This document summarizes the training/evaluation pipeline used by the project.

## Primary Objective

Train and evaluate Layer-1 phishing triage model from URL/host-centric features, with leak-safe splits and reproducible artifacts.

## Main Entry Point

```bash
python -m src.pipeline.run_kaggle_pipeline
```

## Stage Graph

```text
kaggle_ingest
  -> clean
  -> (optional stratified sampling)
  -> enrich --layer1-only
  -> split_leak_safe
  -> leakage_report
  -> train --layer1-only
```

## Important Modules

- `kaggle_ingest.py`: normalize Kaggle labels and schema.
- `clean.py`: dedupe/canonicalize URLs.
- `enrich.py`: feature generation (`layer1_only` mode is the default training path).
- `split_leak_safe.py`: group-aware split by domain keys.
- `train.py`: model training + metrics + artifacts.
- `run_kaggle_pipeline.py`: orchestrates all steps.

Auxiliary:

- `eval_multi_seed.py`: multi-seed stability checks.
- `fp_audit.py` / `phish_audit.py`: audit slices.
- `ai_adjudication_audit.py`: deterministic evidence adjudication audit report generator (historical filename retained).

## Artifacts

- Models: `outputs/models/*` (gitignored)
- Metrics/reports: `outputs/metrics/*`, `outputs/reports/*` (gitignored)
- Processed data: `data/processed/*` (gitignored except placeholders)

## Typical Commands

```bash
# default sampled run
python -m src.pipeline.run_kaggle_pipeline

# quick smoke
python -m src.pipeline.run_kaggle_pipeline --sample-size 5000

# full dataset
python -m src.pipeline.run_kaggle_pipeline --full
```

## Guardrails

- Do not include raw Kaggle files in git.
- Do not include model binaries unless explicitly intended for release.
- Keep training and runtime code separated (`src/pipeline` vs `src/app_v1`).
