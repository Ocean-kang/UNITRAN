#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from unitran.utils.cli import dispatch

COMMANDS = {
    "coco-captions": "unitran.datasets.coco",
    "image-clusters": "unitran.features.patch_store",
}

if __name__ == "__main__":
    dispatch(sys.argv, COMMANDS, "Usage: python tools/query.py <command> [args]")
