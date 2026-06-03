#!/usr/bin/env bash
set -euo pipefail

# Run from any location; the script changes to the UNITRAN repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

IMAGE_NAME="${1:-COCO_train2014_000000000009.jpg}"
OUT_DIR="feature/coco2014_dinov2_vitb14_448"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/visualize_image_cls_attention.py \
  --feature_dir "${OUT_DIR}" \
  --image_name "${IMAGE_NAME}" \
  --device cuda
