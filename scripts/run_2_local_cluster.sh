#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_local_patch_cluster.py --config configs/cluster_local.yaml
