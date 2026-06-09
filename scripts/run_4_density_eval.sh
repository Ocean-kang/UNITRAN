#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_density_eval.py --config configs/local_density_n1000_k20.yaml --stage all
