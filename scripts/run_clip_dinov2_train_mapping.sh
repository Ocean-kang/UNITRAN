#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/train_mapping.py --config configs/train_mapping_clip_dinov2.yaml
