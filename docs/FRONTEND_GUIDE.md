# Frontend Guide (`src/app_v1/frontend.py`)

This guide explains how to run and interpret the Streamlit dashboard.

## Run

From repository root:

```bash
streamlit run src/app_v1/frontend.py
```

Or with Docker:

```bash
docker compose up
```

The frontend shells out to:

```bash
python -m src.app_v1.analyze_dashboard --url <URL>
```

## Inputs

- URL to analyze
- Reinforcement toggle (live capture + org-style checks)
- Layer-1 DNS toggle

## Section-by-Section

### Verdict

- final label (`likely_legitimate`, `uncertain`, `likely_phishing`)
- confidence
- combined score
- reasons trail

### Layer-1 ML

- raw/calibrated phishing probability
- top linear logistic features
- URL canonicalization and model metadata

### Reinforcement

- capture status (`final_url`, redirects, strategy, blocked/error)
- org-style risk and reasons

### Evidence Gaps

Known blind spots where capture/parsing was partial or failed.

### HTML Structure Signals

Compact form/layout-oriented extraction:

- form counts
- password field count
- cross-domain form action
- suspicious phrases

### HTML / DOM Anomaly Review

Richer mismatch reasoning:

- anchor/target mismatches
- form submit behavior
- brand/resource mismatch
- interstitial/wrapper cues
- content-family dampening metadata

### Final Evidence Review

Deterministic evidence adjudication output:

- phishing evidence score
- legitimacy evidence score
- hard blockers
- ambiguity signals
- reason trail

### Host / Path Reasoning

- host identity class
- host legitimacy confidence
- path fit assessment (`plausible`, `unusual_but_possible`, `suspicious`)
- URL decomposition details

## Expected Team Workflow

1. Start from Verdict + Evidence Gaps.
2. Confirm if strong phishing indicators exist (credentials/forms/deceptive host).
3. Use DOM + host/path sections for explainability.
4. Use Final Evidence Review scores/signals as the authoritative verdict basis.

## Troubleshooting

- No output / process failure:
  - run CLI directly: `python -m src.app_v1.analyze_dashboard --url "https://example.com"`
- Incomplete reinforcement:
  - check Docker/network and capture strategy in output.
- No API key is required for standard dashboard operation.
