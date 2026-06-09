#!/usr/bin/env python3
"""Config-driven DINOv2 feature extraction."""

from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_utils import add_config_arg, call_main, cli_args, first_existing, infer_output_dir, load_config, save_resolved_config


def main() -> None:
    parser = add_config_arg("Extract DINOv2 patch/CLS features from COCO images.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = infer_output_dir(cfg, "vision.out_dir", "feature.out_dir")
    save_resolved_config(cfg, out_dir)

    mapping = {
        "coco_root": first_existing(cfg, ["data.coco_root", "coco_root"], "data/mscoco14"),
        "split": first_existing(cfg, ["data.split", "split"], "train2014"),
        "ann_path": first_existing(cfg, ["data.ann_path", "ann_path"], None),
        "image_dir": first_existing(cfg, ["data.image_dir", "image_dir"], None),
        "out_dir": out_dir or "feature/coco2014_dinov2_vitb14_reg_448",
        "model": first_existing(cfg, ["vision.model", "model"], "dinov2_vitb14"),
        "image_size": first_existing(cfg, ["vision.image_size", "image_size"], 448),
        "patch_size": first_existing(cfg, ["vision.patch_size", "patch_size"], 14),
        "batch_size": first_existing(cfg, ["runtime.batch_size", "batch_size"], 32),
        "num_workers": first_existing(cfg, ["runtime.num_workers", "num_workers"], 8),
        "shard_id": first_existing(cfg, ["runtime.shard_id", "shard_id"], 0),
        "num_shards": first_existing(cfg, ["runtime.num_shards", "num_shards"], 1),
        "shard_strategy": first_existing(cfg, ["runtime.shard_strategy", "shard_strategy"], "contiguous"),
        "max_images": first_existing(cfg, ["data.max_images", "max_images"], None),
        "device": first_existing(cfg, ["runtime.device", "device"], "cuda"),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 0),
        "save_dtype": first_existing(cfg, ["vision.save_dtype", "save_dtype"], "float16"),
        "amp_dtype": first_existing(cfg, ["vision.amp_dtype", "amp_dtype"], "float16"),
        "save_image_mean_pt": first_existing(cfg, ["vision.save_image_mean_pt", "save_image_mean_pt"], False),
        "normalize_image_mean": first_existing(cfg, ["vision.normalize_image_mean", "normalize_image_mean"], False),
        "overwrite": first_existing(cfg, ["runtime.overwrite", "overwrite"], False),
        "write_global_meta": first_existing(cfg, ["vision.write_global_meta", "write_global_meta"], True),
    }
    call_main("src.features.dinov2", cli_args(mapping))


if __name__ == "__main__":
    main()
