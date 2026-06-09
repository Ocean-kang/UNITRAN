#!/usr/bin/env python3
"""Config-driven class-level eval prototype extraction."""

from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_utils import add_config_arg, call_main, cli_args, first_existing, infer_output_dir, load_config, save_resolved_config


def main() -> None:
    parser = add_config_arg("Extract class-level vision/text eval prototypes from COCO masks.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = infer_output_dir(cfg, "prototype.out_dir")
    save_resolved_config(cfg, out_dir)

    mapping = {
        "coco_root": first_existing(cfg, ["data.coco_root", "coco_root"], "data/mscoco17"),
        "split": first_existing(cfg, ["data.split", "split"], "val2017"),
        "annotation_type": first_existing(cfg, ["data.annotation_type", "annotation_type"], "auto"),
        "ann_path": first_existing(cfg, ["data.ann_path", "ann_path"], None),
        "image_dir": first_existing(cfg, ["data.image_dir", "image_dir"], None),
        "mask_dir": first_existing(cfg, ["data.mask_dir", "mask_dir"], None),
        "out_dir": out_dir or "feature/eval",
        "out_name": first_existing(cfg, ["prototype.out_name", "out_name"], "eval_prototypes.pt"),
        "model": first_existing(cfg, ["vision.model", "model"], "dinov2_vitb14"),
        "image_size": first_existing(cfg, ["vision.image_size", "image_size"], 448),
        "patch_size": first_existing(cfg, ["vision.patch_size", "patch_size"], 14),
        "mask_to_patch": first_existing(cfg, ["vision.mask_to_patch", "mask_to_patch"], "majority"),
        "normalize_vision": first_existing(cfg, ["vision.normalize_prototypes", "normalize_vision"], True),
        "text_model": first_existing(cfg, ["text.embedding_model", "text_model"], "sentence-transformers/all-mpnet-base-v2"),
        "text_batch_size": first_existing(cfg, ["text.batch_size", "text_batch_size"], 128),
        "batch_size": first_existing(cfg, ["runtime.batch_size", "batch_size"], 16),
        "num_workers": first_existing(cfg, ["runtime.num_workers", "num_workers"], 4),
        "max_images": first_existing(cfg, ["data.max_images", "max_images"], None),
        "min_patch_count": first_existing(cfg, ["prototype.min_patch_count", "min_patch_count"], 1),
        "device": first_existing(cfg, ["runtime.device", "device"], "cuda"),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 0),
        "amp_dtype": first_existing(cfg, ["runtime.amp_dtype", "amp_dtype"], "float16"),
        "overwrite": first_existing(cfg, ["runtime.overwrite", "overwrite"], False),
    }
    call_main("src.evaluation.eval_prototypes", cli_args(mapping))


if __name__ == "__main__":
    main()
