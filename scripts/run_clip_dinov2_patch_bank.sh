#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/extract_clip_dinov2_patch_bank.py --config configs/extract_clip_dinov2_patch_bank.yaml
