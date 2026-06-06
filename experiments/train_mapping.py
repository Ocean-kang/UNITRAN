#!/usr/bin/env python3
"""Config-driven UNITRAN / vec2vec-style mapping training."""

from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_utils import add_config_arg, call_main, cli_args, first_existing, infer_output_dir, load_config, save_resolved_config


def main() -> None:
    parser = add_config_arg("Train UNITRAN mapping from one config file.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = infer_output_dir(cfg, "training.out_dir")
    save_resolved_config(cfg, out_dir)

    mapping = {
        "embedding_dir": first_existing(cfg, ["data.embedding_dir", "embedding_dir"], None),
        "source": first_existing(cfg, ["mapping.source", "source"], "clip"),
        "target": first_existing(cfg, ["mapping.target", "target"], "dinov2"),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 0),
        "num_anchor_runs": first_existing(cfg, ["training.num_anchor_runs", "num_anchor_runs"], 30),
        "num_clusters": first_existing(cfg, ["training.num_clusters", "num_clusters"], 20),
        "qap_runs": first_existing(cfg, ["training.qap_runs", "qap_runs"], 30),
        "anchor_subsample": first_existing(cfg, ["training.anchor_subsample", "anchor_subsample"], 10000),
        "initial_k": first_existing(cfg, ["training.initial_k", "initial_k"], 50),
        "refine_iters": first_existing(cfg, ["training.refine_iters", "refine_iters"], 100),
        "refine_sample": first_existing(cfg, ["training.refine_sample", "refine_sample"], 10000),
        "refine_k": first_existing(cfg, ["training.refine_k", "refine_k"], 50),
        "refine2_clusters": first_existing(cfg, ["training.refine2_clusters", "refine2_clusters"], 500),
        "alpha": first_existing(cfg, ["training.alpha", "alpha"], 0.5),
        "out_dir": out_dir or "outputs/mapping",
    }
    if not mapping["embedding_dir"]:
        raise SystemExit("Missing required config key: data.embedding_dir")
    call_main("src.training.unitran_mapping", cli_args(mapping))


if __name__ == "__main__":
    main()
