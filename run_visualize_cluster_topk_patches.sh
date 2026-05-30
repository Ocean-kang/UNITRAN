#!/usr/bin/env bash
set -euo pipefail

# Run from UNITRAN repository root.
# Requires KMeans assignment outputs:
#   feature/coco2014_dinov2_vitb14_448/kmeans/centroids_k1024_fp32.npy
#   feature/coco2014_dinov2_vitb14_448/assignment/patch_cluster_ids_k1024_uint16.npy
#   feature/coco2014_dinov2_vitb14_448/assignment/patch_cluster_dist_k1024_fp16.npy

FEATURE_DIR="feature/coco2014_dinov2_vitb14_448"

# Debug one cluster first.
python visualize_cluster_topk_patches.py \
  --feature_dir "${FEATURE_DIR}" \
  --k 1024 \
  --topk 50 \
  --clusters 0 \
  --metric cosine \
  --chunk_images 1024 \
  --tile_size 224 \
  --collage_cols 5

# After checking cluster_0000, run all clusters if needed:
# python visualize_cluster_topk_patches.py \
#   --feature_dir "${FEATURE_DIR}" \
#   --k 1024 \
#   --topk 50 \
#   --clusters all \
#   --metric cosine \
#   --chunk_images 1024 \
#   --tile_size 224 \
#   --collage_cols 5
