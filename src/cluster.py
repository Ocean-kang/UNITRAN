"""Cluster-related public entrypoints.

The underlying implementations stay split internally because global streaming KMeans,
per-image local KMeans, text KMeans, and centroid matching have very different runtime
profiles. Users should normally call experiments/run_local_patch_cluster.py or the small
scripts in scripts/ instead of importing these modules directly.
"""

from __future__ import annotations

from src.utils.cli import dispatch

COMMANDS = {
    "patch-global": "src.clustering.stream_kmeans",
    "patch-local": "src.clustering.local_kmeans",
    "caption": "src.clustering.sklearn_kmeans",
    "match-centroids": "src.clustering.centroid_match",
}


def main() -> None:
    import sys

    dispatch(sys.argv, COMMANDS, "Usage: python -m src.cluster <command> [args]")


if __name__ == "__main__":
    main()
