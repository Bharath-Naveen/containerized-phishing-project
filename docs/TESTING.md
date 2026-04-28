# Testing Guide

This project has three testing scopes: unit/logic, pipeline smoke, and runtime app checks.

## 1) Fast local regression (recommended on every change)

```bash
pytest -q
```

Focus:

- feature extraction behavior
- decision policy guardrails
- deterministic evidence adjudication logic
- host/path + DOM anomaly regressions

## 2) Pipeline smoke test

After dataset setup (`docs/DATASET_SETUP.md`):

```bash
python -m src.pipeline.run_kaggle_pipeline --sample-size 5000
```

Verify key outputs:

- `outputs/reports/kaggle_label_audit.json`
- `outputs/reports/leakage_audit.json`
- `outputs/models/layer1_primary.joblib`

## 3) Runtime app checks

### ML-only path

```bash
python -m src.app_v1.analyze_dashboard --url "https://example.com" --no-reinforcement
```

### Full runtime (with reinforcement)

```bash
python -m src.app_v1.analyze_dashboard --url "https://example.com"
```

Expected:

- no crash
- explainable `verdict.reasons`
- sensible `evidence_gaps` if capture is partial

## 4) Streamlit UI check

```bash
streamlit run src/app_v1/frontend.py
```

Validate that each section renders:

- Verdict
- Layer-1 ML
- Reinforcement
- HTML structure
- HTML / DOM anomaly
- Host/path reasoning
- evidence adjudication

## 5) Optional targeted tests

Run only relevant modules while developing:

```bash
pytest tests/test_html_dom_anomaly_signals.py -q
pytest tests/test_host_path_reasoning.py -q
pytest tests/test_ai_adjudicator.py -q
```

## CI Recommendation

On pull requests:

- run `pytest -q`
- run lint/static checks if your team adds them

Nightly or scheduled jobs:

- larger pipeline runs
- capture-heavy runtime checks in Docker
