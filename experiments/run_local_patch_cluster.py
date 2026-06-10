#!/usr/bin/env python3
"""Config-driven per-image local patch clustering."""

from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_utils import add_config_arg, call_main, cli_args, first_existing, infer_output_dir, resolve_config, save_resolved_config


def main() -> None:
    parser = add_config_arg("Run per-image local KMeans over DINOv2 patch tokens.")
    args = parser.parse_args()
    cfg = resolve_config(args.config, args.set)
    out_dir = infer_output_dir(cfg, "cluster.out_dir", "vision.local_feature_dir", "feature.out_dir")
    save_resolved_config(cfg, out_dir)

    mapping = {
        "feature_dir": first_existing(cfg, ["vision.feature_dir", "feature_dir"], "feature/coco2014_dinov2_vitb14_reg_448"),
        "out_dir": out_dir,
        "local_k": first_existing(cfg, ["cluster.local_k", "vision.local_k", "local_k"], 20),
        "local_iters": first_existing(cfg, ["cluster.local_iters", "cluster.num_iters", "local_iters"], 10),
        "metric": first_existing(cfg, ["cluster.metric", "metric"], "cosine"),
        "chunk_images": first_existing(cfg, ["runtime.chunk_images", "cluster.chunk_images", "chunk_images"], 64),
        "device": first_existing(cfg, ["runtime.device", "device"], "cuda:0"),
        "compute_dtype": first_existing(cfg, ["runtime.compute_dtype", "cluster.dtype", "compute_dtype"], "float16"),
        "save_dtype": first_existing(cfg, ["cluster.save_dtype", "save_dtype"], "float16"),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 1234),
        "max_images": first_existing(cfg, ["data.max_images", "max_images"], None),
        "overwrite": first_existing(cfg, ["runtime.overwrite", "overwrite"], False),
        "no_save_pt_outputs": first_existing(cfg, ["cluster.no_save_pt_outputs", "no_save_pt_outputs"], False),
    }
    call_main("src.clustering.local_kmeans", cli_args(mapping))


if __name__ == "__main__":
    main()
