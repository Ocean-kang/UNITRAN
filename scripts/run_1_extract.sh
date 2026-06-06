#!/usr/bin/env bash
set -euo pipefail
python experiments/extract_dinov2_features.py --config configs/extract_dinov2.yaml
