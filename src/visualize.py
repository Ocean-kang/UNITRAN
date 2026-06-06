"""Visualization public entrypoints."""

from __future__ import annotations

from src.utils.cli import dispatch

COMMANDS = {
    "cluster-topk": "src.visualization.cluster_patches",
    "cluster-extremes": "src.visualization.cluster_center_patches",
    "image-cls-attn": "src.visualization.image_cls_attention",
    "image-local-centers": "src.visualization.image_local_centers_attention",
    "embeddings2d": "src.visualization.embedding_2d",
}


def main() -> None:
    import sys

    dispatch(sys.argv, COMMANDS, "Usage: python -m src.visualize <command> [args]")
