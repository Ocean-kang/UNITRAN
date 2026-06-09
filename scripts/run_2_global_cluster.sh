#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_global_patch_cluster.py --config configs/global_cluster_k1024.yaml
