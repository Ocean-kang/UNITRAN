#!/usr/bin/env bash
set -euo pipefail

# Run from UNITRAN repo root after copying the two tools into ./tools/.
# This experiment does NOT use cluster-guided text generation.

COCO_ROOT=${COCO_ROOT:-./coco}
FEATURE_DIR=${FEATURE_DIR:-feature/coco2014_dinov2_vitb14_448}
OUT_DIR=${OUT_DIR:-outputs/text_density_eval}
SAMPLE_JSONL=${SAMPLE_JSONL:-$OUT_DIR/sample_1000.jsonl}
RECAPTION_JSONL=${RECAPTION_JSONL:-$OUT_DIR/recaption_atoms_1000.jsonl}
TEXT_MODEL=${TEXT_MODEL:-sentence-transformers/all-mpnet-base-v2}

mkdir -p "$OUT_DIR"

python tools/build_coco_text_patch_density_eval.py sample \
  --coco_root "$COCO_ROOT" \
  --split train2014 \
  --feature_dir "$FEATURE_DIR" \
  --num_images 1000 \
  --sample_mode random \
  --seed 0 \
  --out_jsonl "$SAMPLE_JSONL"

# Optional: generate recaption atoms with your local Llama3 implementation.
# Uncomment and set paths.
# python tools/generate_coco_llama3_atoms.py \
#   --sample_jsonl "$SAMPLE_JSONL" \
#   --out_jsonl "$RECAPTION_JSONL" \
#   --llama_root ./llama3 \
#   --ckpt_dir /path/to/Meta-Llama-3-8B-Instruct \
#   --tokenizer_path /path/to/Meta-Llama-3-8B-Instruct/tokenizer.model \
#   --max_batch_size 4 \
#   --max_seq_len 2048 \
#   --max_gen_len 512 \
#   --resume

if [[ ! -f "$RECAPTION_JSONL" ]]; then
  echo "[error] Missing $RECAPTION_JSONL" >&2
  echo "Generate it first by uncommenting the Llama3 command above, or pass RECAPTION_JSONL=/path/to/existing_atoms.jsonl" >&2
  exit 1
fi

python tools/build_coco_text_patch_density_eval.py features \
  --sample_jsonl "$SAMPLE_JSONL" \
  --recaption_jsonl "$RECAPTION_JSONL" \
  --feature_dir "$FEATURE_DIR" \
  --text_model "$TEXT_MODEL" \
  --patches_per_image 64 \
  --text_batch_size 128 \
  --device cuda \
  --out_dir "$OUT_DIR"

python tools/build_coco_text_patch_density_eval.py report \
  --out_dir "$OUT_DIR" \
  --max_points 20000 \
  --knn_k 5 \
  --sw_proj 128

# Optional PCA visualization using your existing tool.
python tools/visualize_embeddings_2d.py \
  --input_pt "$OUT_DIR/density_eval_original.pt" \
  --method pca \
  --preprocess l2 \
  --max_points 5000 \
  --out_dir "$OUT_DIR/vis" \
  --prefix original_vs_patch

python tools/visualize_embeddings_2d.py \
  --input_pt "$OUT_DIR/density_eval_expanded.pt" \
  --method pca \
  --preprocess l2 \
  --max_points 5000 \
  --out_dir "$OUT_DIR/vis" \
  --prefix expanded_vs_patch
