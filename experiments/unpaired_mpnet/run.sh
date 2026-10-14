#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
CONFIG=${1:-experiments/unpaired_mpnet/config.yaml}
python experiments/unpaired_mpnet/split_coco.py --config "$CONFIG"
python experiments/unpaired_mpnet/extract_vision.py --config "$CONFIG" --split vision_train
python experiments/unpaired_mpnet/extract_vision.py --config "$CONFIG" --split paired_val
python experiments/unpaired_mpnet/extract_text.py --config "$CONFIG" --split text_train
python experiments/unpaired_mpnet/extract_text.py --config "$CONFIG" --split paired_val
python experiments/unpaired_mpnet/train.py --config "$CONFIG"
python experiments/unpaired_mpnet/evaluate.py --config "$CONFIG"
