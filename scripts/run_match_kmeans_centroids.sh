#!/usr/bin/env bash
set -euo pipefail

# Run from any location; the script changes to the UNITRAN repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FEATURE_DIR="feature/coco2014_dinov2_vitb14_448"
SMALL_K=512
LARGE_K=2048

python tools/match_kmeans_centroids.py \
  --feature_dir "${FEATURE_DIR}" \
  --small_k "${SMALL_K}" \
  --large_k "${LARGE_K}" \
  --overwrite
