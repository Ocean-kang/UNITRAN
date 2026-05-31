#!/usr/bin/env bash
set -euo pipefail

# Run from any location; the script changes to the UNITRAN repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VISION_FEATURE_DIR="feature/coco2014_dinov2_vitb14_448"
CAPTION_KMEANS_DIR="feature/caption_kmeans"
K=512

python tools/visualize_embeddings_2d.py \
  --vision_pt "${VISION_FEATURE_DIR}/kmeans/centroids_k${K}_fp32.npy" \
  --text_pt "${CAPTION_KMEANS_DIR}/centroids_k${K}_fp32.npy" \
  --method pca \
  --preprocess l2 \
  --out_dir outputs/vis \
  --prefix "vision_caption_centroids_k${K}" \
  --title "Vision/caption KMeans centroids"
