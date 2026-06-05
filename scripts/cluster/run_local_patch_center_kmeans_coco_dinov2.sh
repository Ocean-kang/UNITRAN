#!/usr/bin/env bash
set -euo pipefail

# Run from any location; the script changes to the UNITRAN repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RAW_FEATURE_DIR="feature/coco2014_dinov2_vitb14_448"
LOCAL_K=20
LOCAL_FEATURE_DIR="feature/coco2014_dinov2_vitb14_448_local_k${LOCAL_K}"
GLOBAL_K=512

# Stage 1: per image, compress 1024 DINOv2 patch tokens into LOCAL_K local centers.
CUDA_VISIBLE_DEVICES=0 python tools/cluster.py patch-local \
  --feature_dir "${RAW_FEATURE_DIR}" \
  --out_dir "${LOCAL_FEATURE_DIR}" \
  --local_k "${LOCAL_K}" \
  --local_iters 10 \
  --metric cosine \
  --chunk_images 64 \
  --compute_dtype float16 \
  --device cuda:0 \
  --overwrite

# Stage 2: run the existing global streaming KMeans on the local-center feature directory.
CUDA_VISIBLE_DEVICES=0 python tools/cluster.py patch-global \
  --feature_dir "${LOCAL_FEATURE_DIR}" \
  --k "${GLOBAL_K}" \
  --num_iters 20 \
  --metric cosine \
  --chunk_images 512 \
  --compute_dtype float16 \
  --device cuda:0 \
  --assign \
  --save_dist \
  --overwrite

# Optional Stage 3: assign the original 1024 raw patch tokens to the low-noise global centroids.
# Uncomment this block if downstream visualization/query tools still need labels for every original patch token.
# CUDA_VISIBLE_DEVICES=0 python tools/cluster.py patch-global \
#   --feature_dir "${RAW_FEATURE_DIR}" \
#   --k "${GLOBAL_K}" \
#   --metric cosine \
#   --chunk_images 128 \
#   --compute_dtype float16 \
#   --device cuda:0 \
#   --only_assign \
#   --init_centroids "${LOCAL_FEATURE_DIR}/kmeans/centroids_k${GLOBAL_K}_fp32.npy" \
#   --save_dist \
#   --assignment_tag "local_k${LOCAL_K}" \
#   --overwrite
