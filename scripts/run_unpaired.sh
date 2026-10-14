#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/unpaired.py all --config "${1:-configs/unpaired_coco2014.yaml}"
