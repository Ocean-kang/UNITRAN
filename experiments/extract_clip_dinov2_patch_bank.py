#!/usr/bin/env python3
"""Config-driven paired CLIP-V / DINOv2 patch-bank extraction."""

from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_utils import add_config_arg, call_main, cli_args, first_existing, infer_output_dir, resolve_config, save_resolved_config


def main() -> None:
    parser = add_config_arg("Extract paired CLIP-V and DINOv2 patch tokens from COCO images.")
    args = parser.parse_args()
    cfg = resolve_config(args.config, args.set)
    out_dir = infer_output_dir(cfg, "feature.out_dir")
    save_resolved_config(cfg, out_dir)

    mapping = {
        "coco_root": first_existing(cfg, ["data.coco_root", "coco_root"], "data/mscoco14"),
        "split": first_existing(cfg, ["data.split", "split"], "train2014"),
        "ann_path": first_existing(cfg, ["data.ann_path", "ann_path"], None),
        "image_dir": first_existing(cfg, ["data.image_dir", "image_dir"], None),
        "out_dir": out_dir or "feature/clip_dinov2/coco2014_clipL14_dinov2_vitb14_reg_224",
        "clip_model": first_existing(cfg, ["clip.model", "clip_model"], "openai/clip-vit-large-patch14"),
        "dinov2_model": first_existing(cfg, ["vision.model", "dinov2_model"], "dinov2_vitb14_reg"),
        "image_size": first_existing(cfg, ["feature.image_size", "vision.image_size", "image_size"], 224),
        "dinov2_patch_size": first_existing(cfg, ["vision.patch_size", "dinov2_patch_size"], 14),
        "batch_size": first_existing(cfg, ["runtime.batch_size", "batch_size"], 32),
        "num_workers": first_existing(cfg, ["runtime.num_workers", "num_workers"], 8),
        "shard_id": first_existing(cfg, ["runtime.shard_id", "shard_id"], 0),
        "num_shards": first_existing(cfg, ["runtime.num_shards", "num_shards"], 1),
        "shard_strategy": first_existing(cfg, ["runtime.shard_strategy", "shard_strategy"], "contiguous"),
        "max_images": first_existing(cfg, ["data.max_images", "max_images"], None),
        "max_pairs": first_existing(cfg, ["feature.max_pairs", "max_pairs"], 0),
        "device": first_existing(cfg, ["runtime.device", "device"], "cuda"),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 0),
        "save_dtype": first_existing(cfg, ["feature.save_dtype", "save_dtype"], "float16"),
        "amp_dtype": first_existing(cfg, ["feature.amp_dtype", "amp_dtype"], "float16"),
        "normalize_features": first_existing(cfg, ["feature.normalize_features", "normalize_features"], False),
        "overwrite": first_existing(cfg, ["runtime.overwrite", "overwrite"], False),
    }
    call_main("src.features.clip_dinov2_patch_bank", cli_args(mapping))


if __name__ == "__main__":
    main()
