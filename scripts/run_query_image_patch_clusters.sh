#!/usr/bin/env bash
set -euo pipefail

# Run from any location; the script changes to the UNITRAN repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FEATURE_DIR="feature/coco2014_dinov2_vitb14_448"
IMAGE_NAME="COCO_train2014_000000000009.jpg"
KS="all"

python tools/query_image_patch_clusters.py \
  --feature_dir "${FEATURE_DIR}" \
  --image_name "${IMAGE_NAME}" \
  --ks "${KS}"
