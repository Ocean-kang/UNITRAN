#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from unitran.utils.cli import dispatch

COMMANDS = {
    "patch-global": "unitran.clustering.stream_kmeans",
    "patch-local": "unitran.clustering.local_kmeans",
    "caption": "unitran.clustering.sklearn_kmeans",
    "match-centroids": "unitran.clustering.centroid_match",
}

if __name__ == "__main__":
    dispatch(sys.argv, COMMANDS, "Usage: python tools/cluster.py <command> [args]")
