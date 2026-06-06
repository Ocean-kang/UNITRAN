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
        "method": first_existing(cfg, ["mapping.method", "method"], "mini_vec2vec"),
        "embedding_dir": first_existing(cfg, ["data.embedding_dir", "embedding_dir"], None),
        "train_path": first_existing(cfg, ["data.train_path", "train_path"], None),
        "eval_path": first_existing(cfg, ["data.eval_path", "eval_path"], None),
        "source_key": first_existing(cfg, ["data.source_key", "source_key"], "vision_feats"),
        "target_key": first_existing(cfg, ["data.target_key", "target_key"], "text_feats"),
        "source": first_existing(cfg, ["mapping.source", "source"], "clip"),
        "target": first_existing(cfg, ["mapping.target", "target"], "dinov2"),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 0),
        "device": first_existing(cfg, ["training.device", "device"], "cuda"),
        "train_holdout": first_existing(cfg, ["training.train_holdout", "train_holdout"], 8192),
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
        "epochs": first_existing(cfg, ["training.epochs", "epochs"], 20),
        "steps_per_epoch": first_existing(cfg, ["training.steps_per_epoch", "steps_per_epoch"], 0),
        "batch_size": first_existing(cfg, ["training.batch_size", "batch_size"], 256),
        "latent_dim": first_existing(cfg, ["training.latent_dim", "latent_dim"], 768),
        "hidden_dim": first_existing(cfg, ["training.hidden_dim", "hidden_dim"], 1024),
        "depth": first_existing(cfg, ["training.depth", "depth"], 3),
        "discriminator_dim": first_existing(cfg, ["training.discriminator_dim", "discriminator_dim"], 1024),
        "discriminator_depth": first_existing(cfg, ["training.discriminator_depth", "discriminator_depth"], 3),
        "lr": first_existing(cfg, ["training.lr", "lr"], 1e-4),
        "discriminator_lr": first_existing(cfg, ["training.discriminator_lr", "discriminator_lr"], 1e-4),
        "weight_decay": first_existing(cfg, ["training.weight_decay", "weight_decay"], 0.0),
        "max_grad_norm": first_existing(cfg, ["training.max_grad_norm", "max_grad_norm"], 1.0),
        "lambda_rec": first_existing(cfg, ["training.lambda_rec", "lambda_rec"], 5.0),
        "lambda_cc": first_existing(cfg, ["training.lambda_cc", "lambda_cc"], 5.0),
        "lambda_vsp": first_existing(cfg, ["training.lambda_vsp", "lambda_vsp"], 1.0),
        "lambda_adv": first_existing(cfg, ["training.lambda_adv", "lambda_adv"], 1.0),
        "out_dir": out_dir or "outputs/mapping",
    }
    if not mapping["embedding_dir"]:
        raise SystemExit("Missing required config key: data.embedding_dir")
    call_main("src.training.unitran_mapping", cli_args(mapping))


if __name__ == "__main__":
    main()
