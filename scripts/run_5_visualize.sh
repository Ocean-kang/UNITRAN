#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_visualize_cluster.py --config configs/visualize_cluster.yaml
