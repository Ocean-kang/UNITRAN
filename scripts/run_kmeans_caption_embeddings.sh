#!/usr/bin/env bash
set -euo pipefail

# Run from any location; the script changes to the UNITRAN repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# INPUT_PT should contain text_feats, or be a plain .npy array with shape [N, D].
INPUT_PT="feature/talk2dino_avg_self_attn_out_80000.pt"
OUT_DIR="feature/caption_kmeans"
K=512

python tools/cluster_caption_embeddings.py \
  --input_pt "${INPUT_PT}" \
  --text_key text_feats \
  --k "${K}" \
  --preprocess l2 \
  --out_dir "${OUT_DIR}" \
  --overwrite
