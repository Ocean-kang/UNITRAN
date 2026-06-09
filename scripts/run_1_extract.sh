#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/extract_dinov2_features.py --config configs/extract_dinov2.yaml
