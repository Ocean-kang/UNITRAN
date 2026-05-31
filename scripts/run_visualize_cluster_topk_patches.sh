#!/usr/bin/env bash
set -euo pipefail

# Run from any location; the script changes to the UNITRAN repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FEATURE_DIR="feature/coco2014_dinov2_vitb14_448"
K=1024
TOPK=50
CLUSTERS=0

# Requires matching KMeans assignment outputs:
#   feature/coco2014_dinov2_vitb14_448/kmeans/centroids_k${K}_fp32.npy
#   feature/coco2014_dinov2_vitb14_448/assignment/patch_cluster_ids_k${K}_uint16.npy
#   feature/coco2014_dinov2_vitb14_448/assignment/patch_cluster_dist_k${K}_fp16.npy

# Debug one cluster first.
python tools/visualize_cluster_topk_patches.py \
  --feature_dir "${FEATURE_DIR}" \
  --k "${K}" \
  --topk "${TOPK}" \
  --clusters "${CLUSTERS}" \
  --metric cosine \
  --chunk_images 1024 \
  --tile_size 224 \
  --collage_cols 5

# After checking one cluster, set CLUSTERS=all above or run:
# python tools/visualize_cluster_topk_patches.py \
#   --feature_dir "${FEATURE_DIR}" \
#   --k "${K}" \
#   --topk "${TOPK}" \
#   --clusters all \
#   --metric cosine \
#   --chunk_images 1024 \
#   --tile_size 224 \
#   --collage_cols 5
