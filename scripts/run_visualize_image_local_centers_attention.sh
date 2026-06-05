#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RAW_FEATURE_DIR="feature/coco2014_dinov2_vitb14_448"
LOCAL_K=20
LOCAL_FEATURE_DIR="feature/coco2014_dinov2_vitb14_448_local_k${LOCAL_K}"
IMAGE_NAME="COCO_train2014_000000000009.jpg"
LOCAL_CENTERS="all"

python tools/visualize_image_local_centers_attention.py \
  --raw_feature_dir "${RAW_FEATURE_DIR}" \
  --local_feature_dir "${LOCAL_FEATURE_DIR}" \
  --image_name "${IMAGE_NAME}" \
  --local_centers "${LOCAL_CENTERS}" \
  --metric cosine \
  --tile_size 224 \
  --collage_cols 5
