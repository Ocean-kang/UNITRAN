#!/usr/bin/env bash
set -euo pipefail

# Small-sample experiment:
# raw DINOv2 patch tokens -> per-image local K centers -> one text atom per local center.
# Run from any location; the script changes to the UNITRAN repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

COCO_ROOT=${COCO_ROOT:-./coco}
SPLIT=${SPLIT:-train2014}
NUM_IMAGES=${NUM_IMAGES:-1000}
LOCAL_K=${LOCAL_K:-20}
TEXT_MODEL=${TEXT_MODEL:-sentence-transformers/all-mpnet-base-v2}

RAW_FEATURE_DIR=${RAW_FEATURE_DIR:-feature/coco2014_dinov2_vitb14_448_n${NUM_IMAGES}}
LOCAL_FEATURE_DIR=${LOCAL_FEATURE_DIR:-${RAW_FEATURE_DIR}_local_k${LOCAL_K}}
OUT_DIR=${OUT_DIR:-outputs/local_density_eval_n${NUM_IMAGES}_k${LOCAL_K}}
SAMPLE_JSONL=${SAMPLE_JSONL:-${OUT_DIR}/sample_${NUM_IMAGES}.jsonl}
RECAPTION_JSONL=${RECAPTION_JSONL:-${OUT_DIR}/recaption_atoms_${NUM_IMAGES}_k${LOCAL_K}.jsonl}

mkdir -p "${OUT_DIR}"

if [[ ! -f "${RAW_FEATURE_DIR}/images.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python tools/extract.py \
    --coco_root "${COCO_ROOT}" \
    --split "${SPLIT}" \
    --out_dir "${RAW_FEATURE_DIR}" \
    --model dinov2_vitb14 \
    --image_size 448 \
    --batch_size 32 \
    --num_workers 8 \
    --max_images "${NUM_IMAGES}" \
    --save_dtype float16 \
    --amp_dtype float16 \
    --write_global_meta
fi

if [[ ! -f "${LOCAL_FEATURE_DIR}/patch_tokens_shape.json" ]]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python tools/cluster.py patch-local \
    --feature_dir "${RAW_FEATURE_DIR}" \
    --out_dir "${LOCAL_FEATURE_DIR}" \
    --local_k "${LOCAL_K}" \
    --local_iters 10 \
    --metric cosine \
    --chunk_images 64 \
    --compute_dtype float16 \
    --device cuda:0 \
    --overwrite
fi

python tools/evaluate.py local-density sample \
  --coco_root "${COCO_ROOT}" \
  --split "${SPLIT}" \
  --feature_dir "${LOCAL_FEATURE_DIR}" \
  --num_images "${NUM_IMAGES}" \
  --sample_mode first \
  --seed 0 \
  --out_jsonl "${SAMPLE_JSONL}"

# Generate one semantic atom per visual local center.
# Uncomment and set paths if ${RECAPTION_JSONL} does not already exist.
python tools/generate_text_atoms.py \
  --sample_jsonl "${SAMPLE_JSONL}" \
  --out_jsonl "${RECAPTION_JSONL}" \
  --llama_root ./third_party/llama \
  --ckpt_dir ./Meta-Llama-3-8B-Instruct \
  --tokenizer_path ./Meta-Llama-3-8B-Instruct/tokenizer.model \
  --max_batch_size 4 \
  --max_seq_len 2048 \
  --max_gen_len 512 \
  --num_atoms "${LOCAL_K}" \
  --resume

if [[ ! -f "${RECAPTION_JSONL}" ]]; then
  echo "[error] Missing ${RECAPTION_JSONL}" >&2
  echo "Generate it with tools/generate_text_atoms.py --num_atoms ${LOCAL_K}, or pass RECAPTION_JSONL=/path/to/existing.jsonl" >&2
  exit 1
fi

python tools/evaluate.py local-density features \
  --sample_jsonl "${SAMPLE_JSONL}" \
  --recaption_jsonl "${RECAPTION_JSONL}" \
  --local_feature_dir "${LOCAL_FEATURE_DIR}" \
  --atoms_per_image "${LOCAL_K}" \
  --text_model "${TEXT_MODEL}" \
  --text_batch_size 128 \
  --device cuda \
  --out_dir "${OUT_DIR}"

python tools/evaluate.py local-density report \
  --out_dir "${OUT_DIR}" \
  --max_points 20000 \
  --knn_k 5 \
  --sw_proj 128

python tools/visualize.py embeddings2d \
  --input_pt "${OUT_DIR}/local_density_eval_original.pt" \
  --method pca \
  --preprocess l2 \
  --max_points 5000 \
  --out_dir "${OUT_DIR}/vis" \
  --prefix original_vs_local_centers

python tools/visualize.py embeddings2d \
  --input_pt "${OUT_DIR}/local_density_eval_expanded.pt" \
  --method pca \
  --preprocess l2 \
  --max_points 5000 \
  --out_dir "${OUT_DIR}/vis" \
  --prefix expanded_vs_local_centers
