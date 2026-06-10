#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_density_eval.py --config configs/density_local.yaml --stage all
