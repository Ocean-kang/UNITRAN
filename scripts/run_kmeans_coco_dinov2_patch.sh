#!/usr/bin/env bash
set -euo pipefail

# Run from any location; the script changes to the UNITRAN repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Input feature directory should already contain:
#   feature/coco2014_dinov2_vitb14_448/{meta.json,images.jsonl,patch_tokens_shape.json,patch_tokens/*.mmap}

FEATURE_DIR="feature/coco2014_dinov2_vitb14_448"
K=512

# Debug first; uncomment this block for a quick sanity check.
# CUDA_VISIBLE_DEVICES=0 python tools/cluster_coco_dinov2_streaming.py \
#   --feature_dir "${FEATURE_DIR}" \
#   --k 32 \
#   --num_iters 3 \
#   --metric cosine \
#   --chunk_images 32 \
#   --max_images 100 \
#   --assign \
#   --save_dist \
#   --overwrite

# Full run. Prefer a 49GB card.
CUDA_VISIBLE_DEVICES=0 python tools/cluster_coco_dinov2_streaming.py \
  --feature_dir "${FEATURE_DIR}" \
  --k "${K}" \
  --num_iters 20 \
  --metric cosine \
  --chunk_images 128 \
  --compute_dtype float16 \
  --device cuda:0 \
  --assign \
  --save_dist \
  --overwrite
