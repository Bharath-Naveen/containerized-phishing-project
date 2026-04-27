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

## Optional Language Detection (fastText)

The dashboard includes an optional fastText page-language enrichment used as a **contextual signal only**.  
It is not a core phishing verdict driver and does not change model training or thresholds.

Download the language ID model (`lid.176.ftz`) into `./models/`:

```bash
pip install fasttext
python -m src.app_v1.utils.download_models --fasttext
```

Environment variable (optional):

```bash
export PHISH_FASTTEXT_LID_MODEL=/absolute/path/to/lid.176.ftz
```

Behavior:

- If `PHISH_FASTTEXT_LID_MODEL` is set, that path is used.
- If not set, the app falls back to `./models/lid.176.ftz`.
- If no model is found (or `fasttext` is not installed), analysis continues normally and language detection is marked unavailable.

## Legitimacy Rescue Layer

The system includes a generalized legitimacy rescue guard (post-ML, pre-final-verdict) to reduce false positives when structural legitimacy evidence is strong and phishing blockers are absent.

For deployment configuration, trusted-domain registry format, and operational safety notes, see:

- `docs/DEPLOYMENT_NOTES.md`

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

---

## TL;DR (Simple Explanation)

This project is a phishing detection system that analyzes a URL and determines whether it is:

- likely legitimate  
- uncertain  
- likely phishing  

Instead of relying only on a machine learning model, the system combines multiple layers of analysis to make a more reliable and explainable decision.

---

## How the System Works (Simplified)

The system evaluates a website in several steps:

1. **URL-Based ML (Layer 1)**  
   - A machine learning model looks only at the URL and domain features  
   - Produces a fast initial probability  
   - This is only a first signal and can be wrong  

2. **Live Page Analysis (Reinforcement Layer)**  
   - The system loads the page safely  
   - Checks for basic behavioral and branding signals  

3. **HTML / DOM Analysis**  
   - Examines page structure  
   - Looks for:
     - login forms  
     - suspicious redirects  
     - fake wrapper/interstitial pages  

4. **Host and Path Reasoning**  
   - Evaluates the domain and URL path  
   - Determines if the URL structure is normal or suspicious  

5. **AI Adjudication (Optional)**  
   - Used only for borderline cases  
   - Reviews structured evidence (not raw HTML)  
   - Provides a bounded adjustment to the decision  

6. **Final Safety Guardrail**  
   - If the page shows **no phishing behavior** (no credential capture, no impersonation, no deceptive redirects),  
     the system forces a **likely legitimate** decision  
   - This prevents false positives from the ML model  

---

## How to Interpret the Dashboard

When you analyze a URL in the frontend, each section represents a part of the decision:

- **Verdict**  
  Final classification, confidence level, and combined score  

- **Layer-1 ML**  
  The model’s prediction based only on the URL  

- **Reinforcement**  
  Results from loading and observing the live page  

- **HTML Structure Signals**  
  Summary of page structure (forms, links, layout patterns)  

- **DOM Anomaly Review**  
  Detection of suspicious patterns like fake wrappers or mismatched links  

- **Host / Path Reasoning**  
  Whether the domain and URL path look legitimate or suspicious  

- **AI Adjudication**  
  Optional AI-based reasoning used only for uncertain cases  

- **Evidence Gaps**  
  Missing signals due to capture issues or blocked content  

---

## Key Design Idea

The system is intentionally designed so that:

> The final decision is based on **evidence**, not just the ML model.

This reduces false positives on legitimate websites while still detecting phishing pages that show real malicious behavior.
