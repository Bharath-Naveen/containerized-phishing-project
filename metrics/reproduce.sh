#!/usr/bin/env bash
# Regenerate every number in metrics/VERIFIED_METRICS.{json,md}.
#
# Prerequisites (not committed to git, see docs/DATASET_SETUP.md):
#   data/raw/kaggle/phishing_and_legitimate_urls.csv          Kaggle harisudhan411/phishing-and-legitimate-urls
#   outputs/models/*.joblib                                   shipped models (steps 0, 4, 5)
#   data/processed/retrain_with_fresh_{train,test}_20260427_234728.csv   saved split of the shipped models (step 5)
#   phishing_dataset/dataset_{full,test_recent}.csv          PhishStats + curated official URLs (steps 4, 5)
#
# Everything intermediate is written to metrics/_work/ (gitignored). outputs/ and data/processed/ are not modified.
# Runtime on 2 vCPU: about 1.5 hours (step 4 dashboard calls ~45 min, step 6 full run ~20 min).
#
# Usage:  bash metrics/reproduce.sh            (from the repo root, inside the venv from requirements.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
pip install -q "pytest-cov>=5" >/dev/null 2>&1 || true

python metrics/scripts/00_tests.py            # pytest counts + coverage
python metrics/scripts/01_pipeline.py         # documented 50K default pipeline, end to end, timed
python metrics/scripts/02_evaluate_layer1.py  # held-out metrics, dummy baseline, bootstrap CI, 5-fold grouped CV
python metrics/scripts/03_ablations.py        # scheme-artifact and eval-contamination checks
python metrics/scripts/04_suites_and_fp.py    # dashboard verdicts on suites / legit lists, rescue on vs off, latency
python metrics/scripts/05_deployed_model.py   # shipped model on its saved held-out split, official URLs, calibrator
python metrics/scripts/05b_deployed_model_leakfix_eval.py  # shipped model re-scored on leak-free rows with app-path features
python metrics/scripts/06_full_run.py         # full ~797K-row run of the documented pipeline (about 20 min incl. bootstrap)
python metrics/scripts/build_report.py        # writes metrics/VERIFIED_METRICS.json and .md
