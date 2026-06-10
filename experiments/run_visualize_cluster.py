#!/usr/bin/env python3
"""Config-driven visualization entrypoint."""

from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_utils import add_config_arg, call_main, cli_args, first_existing, infer_output_dir, resolve_config, save_resolved_config

MODULES = {
    "cluster_topk": "src.visualization.cluster_patches",
    "cluster_extremes": "src.visualization.cluster_center_patches",
    "image_cls_attention": "src.visualization.image_cls_attention",
    "image_local_centers": "src.visualization.image_local_centers_attention",
    "embedding_2d": "src.visualization.embedding_2d",
}


def _cluster_args(cfg: dict, out_dir: str | None) -> dict:
    return {
        "feature_dir": first_existing(cfg, ["vision.feature_dir", "feature_dir"], "feature/vision/coco2014_dinov2_vitb14_reg_448"),
        "k": first_existing(cfg, ["cluster.k", "k"], 1024),
        "topk": first_existing(cfg, ["visualization.topk", "topk"], 50),
        "clusters": first_existing(cfg, ["visualization.clusters", "clusters"], "all"),
        "metric": first_existing(cfg, ["visualization.metric", "metric"], "cosine"),
        "chunk_images": first_existing(cfg, ["runtime.chunk_images", "chunk_images"], 1024),
        "tile_size": first_existing(cfg, ["visualization.tile_size", "tile_size"], 224),
        "collage_cols": first_existing(cfg, ["visualization.collage_cols", "collage_cols"], 5),
        "overlay_alpha": first_existing(cfg, ["visualization.overlay_alpha", "overlay_alpha"], 0.45),
        "image_root": first_existing(cfg, ["data.image_root", "image_root"], None),
        "centroids": first_existing(cfg, ["cluster.centroids", "centroids"], None),
        "cluster_ids": first_existing(cfg, ["cluster.cluster_ids", "cluster_ids"], None),
        "cluster_dist": first_existing(cfg, ["cluster.cluster_dist", "cluster_dist"], None),
        "out_dir": out_dir,
        "save_each": first_existing(cfg, ["visualization.save_each", "save_each"], False),
    }


def _image_local_args(cfg: dict, out_dir: str | None) -> dict:
    return {
        "local_feature_dir": first_existing(cfg, ["vision.local_feature_dir", "local_feature_dir"], "feature/vision/coco2014_dinov2_vitb14_reg_448_local_k20"),
        "raw_feature_dir": first_existing(cfg, ["vision.raw_feature_dir", "raw_feature_dir"], None),
        "image_name": first_existing(cfg, ["visualization.image_name", "image_name"], None),
        "image_index": first_existing(cfg, ["visualization.image_index", "image_index"], None),
        "local_centers": first_existing(cfg, ["visualization.local_centers", "local_centers"], "all"),
        "metric": first_existing(cfg, ["visualization.metric", "metric"], "cosine"),
        "image_root": first_existing(cfg, ["data.image_root", "image_root"], None),
        "tile_size": first_existing(cfg, ["visualization.tile_size", "tile_size"], 224),
        "collage_cols": first_existing(cfg, ["visualization.collage_cols", "collage_cols"], 5),
        "overlay_alpha": first_existing(cfg, ["visualization.overlay_alpha", "overlay_alpha"], 0.45),
        "out_dir": out_dir,
    }


def _image_cls_args(cfg: dict, out_dir: str | None) -> dict:
    return {
        "feature_dir": first_existing(cfg, ["vision.feature_dir", "feature_dir"], "feature/vision/coco2014_dinov2_vitb14_reg_448"),
        "image_name": first_existing(cfg, ["visualization.image_name", "image_name"], None),
        "image_index": first_existing(cfg, ["visualization.image_index", "image_index"], None),
        "image_root": first_existing(cfg, ["data.image_root", "image_root"], None),
        "model": first_existing(cfg, ["vision.model", "model"], None),
        "image_size": first_existing(cfg, ["vision.image_size", "image_size"], None),
        "device": first_existing(cfg, ["runtime.device", "device"], "cuda"),
        "overlay_alpha": first_existing(cfg, ["visualization.overlay_alpha", "overlay_alpha"], 0.45),
        "out_dir": out_dir,
    }


def _embedding_args(cfg: dict, out_dir: str | None) -> dict:
    return {
        "input_pt": first_existing(cfg, ["embedding.input_pt", "input_pt"], None),
        "vision_pt": first_existing(cfg, ["embedding.vision_pt", "vision_pt"], None),
        "text_pt": first_existing(cfg, ["embedding.text_pt", "text_pt"], None),
        "vision_key": first_existing(cfg, ["embedding.vision_key", "vision_key"], "vision_feats"),
        "text_key": first_existing(cfg, ["embedding.text_key", "text_key"], "text_feats"),
        "W": first_existing(cfg, ["embedding.W", "W"], None),
        "method": first_existing(cfg, ["visualization.method", "method"], "pca"),
        "preprocess": first_existing(cfg, ["visualization.preprocess", "preprocess"], "l2"),
        "max_points": first_existing(cfg, ["visualization.max_points", "max_points"], 5000),
        "point_size": first_existing(cfg, ["visualization.point_size", "point_size"], 8.0),
        "dpi": first_existing(cfg, ["visualization.dpi", "dpi"], 300),
        "seed": first_existing(cfg, ["experiment.seed", "seed"], 0),
        "out_dir": out_dir or "outputs/vis",
        "prefix": first_existing(cfg, ["visualization.prefix", "prefix"], "embeddings_2d"),
        "title": first_existing(cfg, ["visualization.title", "title"], "Text/Vision embeddings"),
    }


def main() -> None:
    parser = add_config_arg("Run cluster/image/embedding visualization from one config.")
    args = parser.parse_args()
    cfg = resolve_config(args.config, args.set)
    mode = first_existing(cfg, ["visualization.mode", "mode"], "cluster_topk")
    if mode not in MODULES:
        raise SystemExit(f"Unknown visualization.mode={mode!r}. Available: {', '.join(MODULES)}")
    out_dir = infer_output_dir(cfg, "visualization.out_dir")
    save_resolved_config(cfg, out_dir)

    if mode in {"cluster_topk", "cluster_extremes"}:
        mapping = _cluster_args(cfg, out_dir)
    elif mode == "image_cls_attention":
        mapping = _image_cls_args(cfg, out_dir)
    elif mode == "image_local_centers":
        mapping = _image_local_args(cfg, out_dir)
    else:
        mapping = _embedding_args(cfg, out_dir)
    call_main(MODULES[mode], cli_args(mapping))


if __name__ == "__main__":
    main()
