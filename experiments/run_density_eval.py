#!/usr/bin/env python3
"""Config-driven text/vision density evaluation.

Use evaluation.kind: local for local-center vs text atoms, or patch for raw patch-token density.
"""

from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_utils import add_config_arg, call_main, cli_args, first_existing, infer_output_dir, load_config, save_resolved_config


def _module(cfg: dict) -> str:
    kind = first_existing(cfg, ["evaluation.kind", "kind"], "local")
    if kind in {"local", "local_density"}:
        return "src.evaluation.local_density"
    if kind in {"patch", "text_density", "raw_patch"}:
        return "src.evaluation.text_density"
    raise SystemExit(f"Unknown evaluation.kind={kind!r}. Use 'local' or 'patch'.")


def _sample_args(cfg: dict, out_dir: str | None, local: bool) -> dict:
    default_feature = "feature/coco2014_dinov2_vitb14_448_local_k20" if local else "feature/coco2014_dinov2_vitb14_448"
    sample_path = first_existing(cfg, ["sample.out_jsonl", "evaluation.sample_jsonl"], None)
    if sample_path is None and out_dir:
        n = first_existing(cfg, ["data.num_images", "num_images"], 1000)
        sample_path = f"{out_dir}/sample_{n}.jsonl"
    return {
        "coco_root": first_existing(cfg, ["data.coco_root", "coco_root"], "./coco"),
        "split": first_existing(cfg, ["data.split", "split"], "train2014"),
        "ann_path": first_existing(cfg, ["data.ann_path", "ann_path"], None),
        "feature_dir": first_existing(cfg, ["vision.local_feature_dir" if local else "vision.feature_dir", "feature_dir"], default_feature),
        "num_images": first_existing(cfg, ["data.num_images", "num_images"], 1000),
        "max_captions": first_existing(cfg, ["text.max_captions", "max_captions"], 5),
        "sample_mode": first_existing(cfg, ["sample.mode", "sample_mode"], "first" if local else "random"),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 0),
        "out_jsonl": sample_path,
    }


def _feature_args(cfg: dict, out_dir: str | None, local: bool) -> dict:
    default_feature = "feature/coco2014_dinov2_vitb14_448_local_k20" if local else "feature/coco2014_dinov2_vitb14_448"
    sample_path = first_existing(cfg, ["evaluation.sample_jsonl", "sample.out_jsonl"], None)
    if sample_path is None and out_dir:
        n = first_existing(cfg, ["data.num_images", "num_images"], 1000)
        sample_path = f"{out_dir}/sample_{n}.jsonl"
    common = {
        "sample_jsonl": sample_path,
        "recaption_jsonl": first_existing(cfg, ["text.recaption_jsonl", "recaption_jsonl"], None),
        "text_model": first_existing(cfg, ["text.embedding_model", "text_model"], "sentence-transformers/all-mpnet-base-v2"),
        "text_batch_size": first_existing(cfg, ["runtime.text_batch_size", "text_batch_size"], 128),
        "device": first_existing(cfg, ["runtime.device", "device"], "cuda"),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 0),
        "out_dir": out_dir or ("outputs/local_density_eval" if local else "outputs/text_density_eval"),
    }
    if local:
        common.update({
            "local_feature_dir": first_existing(cfg, ["vision.local_feature_dir", "local_feature_dir"], default_feature),
            "atoms_per_image": first_existing(cfg, ["text.atoms_per_image", "atoms_per_image"], 0),
        })
    else:
        common.update({
            "feature_dir": first_existing(cfg, ["vision.feature_dir", "feature_dir"], default_feature),
            "patches_per_image": first_existing(cfg, ["vision.patches_per_image", "patches_per_image"], 64),
        })
    return common


def _report_args(cfg: dict, out_dir: str | None) -> dict:
    return {
        "out_dir": out_dir or first_existing(cfg, ["evaluation.out_dir"], "outputs/local_density_eval"),
        "max_points": first_existing(cfg, ["evaluation.max_points", "max_points"], 20000),
        "knn_k": first_existing(cfg, ["evaluation.knn_k", "knn_k"], 5),
        "sw_proj": first_existing(cfg, ["evaluation.sw_proj", "sw_proj"], 128),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 0),
    }


def main() -> None:
    parser = add_config_arg("Run density evaluation from one config file.")
    parser.add_argument("--stage", choices=["sample", "features", "report", "all"], default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = infer_output_dir(cfg, "evaluation.out_dir")
    save_resolved_config(cfg, out_dir)

    module = _module(cfg)
    local = module.endswith("local_density")
    stages = ["sample", "features", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "sample":
            mapping = _sample_args(cfg, out_dir, local)
        elif stage == "features":
            mapping = _feature_args(cfg, out_dir, local)
        else:
            mapping = _report_args(cfg, out_dir)
        call_main(module, [stage, *cli_args(mapping)])


if __name__ == "__main__":
    main()
