#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/raw"
cp "$ROOT/archive/sample_data/challenge"/*.txt "$ROOT/data/raw/"
echo "Copied challenge samples to $ROOT/data/raw"
