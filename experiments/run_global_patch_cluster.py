#!/usr/bin/env python3
"""Config-driven global KMeans over all DINOv2 patch tokens."""

from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_utils import add_config_arg, call_main, cli_args, first_existing, infer_output_dir, load_config, save_resolved_config


def main() -> None:
    parser = add_config_arg("Run global streaming KMeans over all DINOv2 patch tokens.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    feature_dir = first_existing(cfg, ["vision.feature_dir", "feature_dir"], "feature/coco2014_dinov2_vitb14_reg_448")
    out_dir = infer_output_dir(cfg, "cluster.output_dir", "vision.feature_dir", "feature_dir") or feature_dir
    save_resolved_config(cfg, out_dir)

    mapping = {
        "feature_dir": feature_dir,
        "k": first_existing(cfg, ["cluster.k", "k"], 1024),
        "num_iters": first_existing(cfg, ["cluster.num_iters", "num_iters"], 20),
        "metric": first_existing(cfg, ["cluster.metric", "metric"], "cosine"),
        "chunk_images": first_existing(cfg, ["runtime.chunk_images", "chunk_images"], 128),
        "distance_chunk": first_existing(cfg, ["runtime.distance_chunk", "distance_chunk"], 0),
        "device": first_existing(cfg, ["runtime.device", "device"], "cuda:0"),
        "compute_dtype": first_existing(cfg, ["runtime.compute_dtype", "compute_dtype"], "float16"),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 1234),
        "init_centroids": first_existing(cfg, ["cluster.init_centroids", "init_centroids"], None),
        "empty_cluster_policy": first_existing(cfg, ["cluster.empty_cluster_policy", "empty_cluster_policy"], "keep_old"),
        "max_images": first_existing(cfg, ["data.max_images", "max_images"], None),
        "assign": first_existing(cfg, ["cluster.assign", "assign"], True),
        "only_assign": first_existing(cfg, ["cluster.only_assign", "only_assign"], False),
        "save_dist": first_existing(cfg, ["cluster.save_dist", "save_dist"], True),
        "assignment_tag": first_existing(cfg, ["cluster.assignment_tag", "assignment_tag"], ""),
        "overwrite": first_existing(cfg, ["runtime.overwrite", "overwrite"], False),
    }
    call_main("src.clustering.stream_kmeans", cli_args(mapping))


if __name__ == "__main__":
    main()
