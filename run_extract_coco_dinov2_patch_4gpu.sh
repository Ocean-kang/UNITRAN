#!/usr/bin/env bash
set -euo pipefail

# Run from UNITRAN repository root.
# COCO expected layout:
#   ./coco/train2014/*.jpg
#   ./coco/annotations/captions_train2014.json  (optional but recommended)

OUT_DIR="feature/coco2014_dinov2_vitb14_448"
COMMON_ARGS="--coco_root ./coco --split train2014 --out_dir ${OUT_DIR} --model dinov2_vitb14 --image_size 448 --num_shards 4 --shard_strategy contiguous --save_dtype float16 --amp_dtype float16"

# Two 49GB GPUs: larger batch; two 24GB GPUs: smaller batch.
CUDA_VISIBLE_DEVICES=0 python extract_coco_dinov2_patch.py ${COMMON_ARGS} --shard_id 0 --batch_size 64 --num_workers 8 --write_global_meta &
CUDA_VISIBLE_DEVICES=1 python extract_coco_dinov2_patch.py ${COMMON_ARGS} --shard_id 1 --batch_size 64 --num_workers 8 &
CUDA_VISIBLE_DEVICES=2 python extract_coco_dinov2_patch.py ${COMMON_ARGS} --shard_id 2 --batch_size 32 --num_workers 8 &
CUDA_VISIBLE_DEVICES=3 python extract_coco_dinov2_patch.py ${COMMON_ARGS} --shard_id 3 --batch_size 32 --num_workers 8 &
wait

echo "All shards finished. Output: ${OUT_DIR}"
