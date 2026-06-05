#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RAW_FEATURE_DIR="feature/coco2014_dinov2_vitb14_448"
LOCAL_K=20
LOCAL_FEATURE_DIR="feature/coco2014_dinov2_vitb14_448_local_k${LOCAL_K}"
GLOBAL_K=512
TOPK=50
CLUSTERS=0,1,2,3,4,5,6,7,8,9,10
ASSIGN_TAG="local_k${LOCAL_K}"

# This launcher expects the optional raw-token assignment in
# scripts/run_local_patch_center_kmeans_coco_dinov2.sh Stage 3 to have been run.
python tools/visualize.py cluster-topk \
  --feature_dir "${RAW_FEATURE_DIR}" \
  --k "${GLOBAL_K}" \
  --topk "${TOPK}" \
  --clusters "${CLUSTERS}" \
  --metric cosine \
  --centroids "${LOCAL_FEATURE_DIR}/kmeans/centroids_k${GLOBAL_K}_fp32.npy" \
  --cluster_ids "${RAW_FEATURE_DIR}/assignment/patch_cluster_ids_k${GLOBAL_K}_${ASSIGN_TAG}_uint16.npy" \
  --cluster_dist "${RAW_FEATURE_DIR}/assignment/patch_cluster_dist_k${GLOBAL_K}_${ASSIGN_TAG}_fp16.npy" \
  --out_dir "${RAW_FEATURE_DIR}/vis/local_global_topk_patches_k${GLOBAL_K}_${ASSIGN_TAG}_top${TOPK}" \
  --chunk_images 1024 \
  --tile_size 224 \
  --collage_cols 5
