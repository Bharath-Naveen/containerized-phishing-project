# Containerized Phishing Detection Dashboard

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
$env:PYTHONPATH = "$PWD"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app_v1/frontend.py
```

CLI one-off analysis:

```powershell
python -m src.app_v1.analyze_dashboard --url "https://example.com"
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

Every number here was produced by re-running the code (seed 42) and is saved with its command, commit, data hash and environment in [metrics/VERIFIED_METRICS.md](metrics/VERIFIED_METRICS.md). Regenerate with `bash metrics/reproduce.sh`.

Layer-1 URL model, full Kaggle dataset (796,446 deduplicated URLs plus 837 curated legitimate URLs), held-out test of 140,057 rows from domains never seen in training, threshold 0.5:

| Model | F1 | ROC-AUC | Precision | False-positive rate |
|---|---|---|---|---|
| Majority-class baseline | 0.000 | 0.500 | 0.000 | 0.0% |
| LightGBM | 0.881 | 0.954 | 0.921 | 7.3% |
| Random Forest | 0.864 | 0.940 | 0.936 | 5.6% |

- Tests: 253 of 253 pass; 52% line coverage of `src/`.
- Measured limits: on 298 curated official brand URLs, the Layer-1 models alone flag 35% to 54% as phishing. Without live capture, the adjudication layer returns `uncertain` for all 298 (none `likely_phishing`), and it returned `likely_legitimate` for none of the roughly 1,600 URLs tested that way.
- The models currently in `outputs/models` came from `retrain_with_fresh.py`, whose split did not group scheme-less URLs by domain. See the metrics file before quoting those numbers.

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

## Suggested Repo Layout (Deployment-Ready)

```text
.
├── src/                      # Runtime + pipeline code
├── tests/                    # Regression tests
├── docs/                     # Design and operational docs
├── data/
│   ├── official_domains.json # curated trust-prior registry (kept)
│   ├── raw/ interim/ processed/ (runtime/training data, gitignored)
├── models/                   # optional demo artifacts
├── Dockerfile
├── docker-compose.yml
└── README.md
```

Generated runtime artifacts should remain untracked:

- `captures/`
- `outputs/fresh_retrain_runs/`
- `outputs/reports/` debug artifacts
- `data/processed/`
- temporary CSV/JSONL exports

## Useful Commands

```bash
# targeted regression set used for deployment checks
pytest tests/test_evidence_adjudication_layer.py tests/test_hosting_domain_trust_layer.py tests/test_ml_model_agreement.py tests/test_behavior_signals.py -q
```

# This project is under review and will be rebuild after planning
