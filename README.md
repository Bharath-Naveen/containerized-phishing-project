# Containerized Phishing Detection Dashboard

[![CI](https://github.com/Bharath-Naveen/containerized-phishing-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Bharath-Naveen/containerized-phishing-project/actions/workflows/ci.yml)

Layered phishing detection dashboard combining ML triage and deterministic evidence review for explainable, deployment-ready decisions.

## Project Goal

Detect phishing reliably while reducing false positives on legitimate modern websites by combining:

- fast URL/domain ML scoring,
- live browser evidence collection,
- structural and behavior analysis,
- deterministic Evidence Adjudication Layer (EAL).

The runtime path is deterministic. AI adjudication is removed/disabled.

## Current Architecture

1. **Layer 1: ML + Model Agreement**
   - Primary Layer-1 model trained on large URL/host feature data (~500k scale training setup).
   - Optional witness models (`logistic_regression`, `random_forest`, `xgboost`, `lightgbm`) provide agreement/disagreement signals.
   - Primary model remains authoritative for ML probability.

2. **Layer 2: Live Capture (Playwright)**
   - Final URL + redirect chain collection.
   - Form target capture (same-domain vs cross-domain).
   - TLS/browser security state (`https`, cert errors, insecure/mixed content indicators).

3. **Layer 3: HTML/DOM + Behavior Signals**
   - DOM/form/link structure signals.
   - Wrapper/interstitial, credential harvester, and suspicious-page patterns.
   - JS/network behavior heuristics (including exfiltration suspicion).

4. **Brand/Trust Context**
   - Brand-domain coherence (deterministic NLP-style matching).
   - Official domain trust-prior (`data/official_domains.json`) used as a weak trust anchor, not a whitelist.

5. **Evidence Adjudication Layer (EAL)**
   - Deterministic phishing/legitimacy/ambiguity scoring.
   - Hard blockers for high-risk corroborated patterns.
   - Conservative conflict handling (`uncertain` when evidence disagrees).

## AI Status

- AI/OpenAI adjudication is not used in deployment runtime.
- No OpenAI key is required for dashboard/CLI operation.

## Setup

### Local Python run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .            # installs the `phishguard` command
phishguard serve            # Streamlit dashboard on http://localhost:8501 (uses the verified model in models/layer1/)
```

One command for everything:

```powershell
phishguard analyze --url "https://example.com"   # score one URL (add --no-reinforcement for ML-only)
phishguard train                                 # train on a 50K stratified Kaggle sample (--full for all rows)
phishguard evaluate                              # every verified metric + model card (after phishguard train)
phishguard evaluate-live                         # full system with live page capture (needs internet; runs in GitHub Actions)
phishguard deploy                                # ship the latest trained model bundle
```

### Docker run

```bash
docker compose build
docker compose up
```

Then open [http://localhost:8501](http://localhost:8501).

## Example URLs to Validate Behavior

- LinkedIn official profile/login (`linkedin.com`) -> `likely_legitimate` or `uncertain`
- Virgin Atlantic (`virginatlantic.com`) -> `likely_legitimate` or `uncertain`
- Coursera (`coursera.org`) -> `likely_legitimate` or `uncertain`
- Suspicious Weebly/user-hosted sample -> `likely_phishing`
- Vercel brand clone (`paypal-login.vercel.app`, `netflix-update-payment-details.vercel.app`) -> `likely_phishing`

## How to Interpret Verdicts

- `likely_phishing`: corroborated high-risk phishing evidence.
- `uncertain`: evidence conflict or insufficient corroboration; manual review recommended.
- `likely_legitimate`: strong legitimacy evidence with no high-risk phishing blockers.

## Results

Every number here comes from `phishguard evaluate` (seed 42) and is saved with its command, commit, data hash and environment in [metrics/VERIFIED_METRICS.md](metrics/VERIFIED_METRICS.md). Regenerate everything with `bash metrics/reproduce.sh` (about 25 minutes on 2 CPU cores). Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

Layer-1 URL model, trained on the full Kaggle dataset (794,563 deduplicated URLs) and tested on 147,702 URLs from domains never seen in training (threshold 0.5):

| Model | F1 (95% CI) | ROC-AUC | Precision | Recall | False-positive rate |
|---|---|---|---|---|---|
| Majority-class baseline | 0.000 | 0.500 | 0.000 | 0.000 | 0.0% |
| **XGBoost (selected on validation data)** | 0.801 (0.799 to 0.803) | 0.880 | 0.772 | 0.832 | 21.3% |
| Random Forest | 0.793 | 0.884 | 0.784 | 0.802 | 19.2% |
| LightGBM | 0.803 | 0.877 | 0.764 | 0.847 | 22.7% |

- Tests: 365 of 365 pass; 54.0% line coverage of `src/phishguard`.
- Real-world checks (never in training): the URL model alone flags 40.4% of 136 official brand URLs and 81.0% of 284 PhishStats phishing URLs. The adjudication layer turns none of the official URLs into a phishing verdict (all go to `uncertain` without live capture).
- These numbers replace an earlier audit of the original code, which had a leaky split; that record is in `metrics/audit_baseline/`.

## Known Limitations

- URL-based ML can over-alert on modern JS-heavy sites.
- Live capture can be affected by anti-bot systems, geofencing, or headless blocking.
- Official trust-prior is not a whitelist and does not auto-trust suspicious behavior.
- `uncertain` is intentional for conflict cases where deterministic evidence disagrees.

## Deployment Notes

- **Laptop/local run**: only users on the same machine (or LAN if allowed) can access the dashboard.
- **Public access**: deploy to a hosted environment (VPS/cloud), e.g. AWS, GCP, Azure, Render, Fly.io, Railway, etc.
- Expose port `8501` and secure access (HTTPS + auth/network controls) before sharing publicly.
- Never store secrets in repo files or compose files.

## Repo Layout

```text
.
├── src/phishguard/
│   ├── urls/          # URL parsing + the one canonical/scheme-neutral normalizer
│   ├── data/          # Kaggle ingest, clean, sample, evaluation sets, enrich, domain-grouped split
│   ├── features/      # Layer-1 URL/host features
│   ├── models/        # training (validated selection, model bundle), deploy
│   ├── pipelines/     # end-to-end training pipelines (phishguard train)
│   ├── evaluation/    # URL suites, false-positive / phishing audits, reports
│   ├── app/           # dashboard, EAL, capture, signals, Streamlit UI
│   ├── guards.py      # checks that stop leaky splits and train/serve skew
│   ├── config.py      # seed and defaults
│   └── cli.py         # the phishguard command
├── tests/             # unit, regression and golden-output tests
├── metrics/           # verified, reproducible metrics (see Results)
├── docs/rebuild/      # how it works + rebuild log
├── models/layer1/     # the verified model the app ships (MANIFEST.json has hashes and provenance)
├── data/              # evaluation sets + registries (tracked); raw/processed data (gitignored)
├── .github/workflows/ # CI (tests + Docker build) and the live-capture evaluation
├── archive/legacy/    # retired AI adjudication path and old scripts
├── Dockerfile, docker-compose.yml, pyproject.toml
└── README.md
```

Generated runtime artifacts should remain untracked:

- `captures/`
- `outputs/fresh_retrain_runs/`
- `outputs/reports/` debug artifacts
- `data/processed/`
- temporary CSV/JSONL exports

## Tests

```bash
pytest            # full suite, including golden-output tests that pin the dashboard's behavior
```

## Rebuild in progress

This project is being rebuilt for verified, reproducible results. What changed and why: [docs/rebuild/REBUILD_LOG.md](docs/rebuild/REBUILD_LOG.md). How the system works: [docs/rebuild/HOW_IT_WORKS.md](docs/rebuild/HOW_IT_WORKS.md).
