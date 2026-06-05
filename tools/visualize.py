#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from unitran.utils.cli import dispatch

COMMANDS = {
    "cluster-topk": "unitran.visualization.cluster_patches",
    "cluster-extremes": "unitran.visualization.cluster_center_patches",
    "image-cls-attn": "unitran.visualization.image_cls_attention",
    "image-local-centers": "unitran.visualization.image_local_centers_attention",
    "embeddings2d": "unitran.visualization.embedding_2d",
}

if __name__ == "__main__":
    dispatch(sys.argv, COMMANDS, "Usage: python tools/visualize.py <command> [args]")
