#!/usr/bin/env bash
# Regenerate metrics/VERIFIED_METRICS.md, metrics/results/evaluation.json and docs/MODEL_CARD.md.
#
# Needs: the Kaggle CSV in data/raw/kaggle/ (see docs/DATASET_SETUP.md), Python deps from
# requirements.txt, and `pip install -e .` for the phishguard command. About 30 minutes on 2 vCPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=42
pip install -q "pytest-cov>=5" >/dev/null 2>&1 || true
phishguard train --full --seed 42 --no-enrich-resume "$@"
phishguard evaluate --with-tests
