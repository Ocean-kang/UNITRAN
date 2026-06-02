#!/usr/bin/env python3
"""
Visualize the most similar and least similar assigned patch tokens for each KMeans cluster center.

This script reuses the rendering utilities from `tools/visualize_cluster_topk_patches.py`.
For each requested cluster, it mines from saved assignment files:
  1. top-k assigned patch tokens closest to the cluster center;
  2. top-k assigned patch tokens farthest from the cluster center.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from visualize_cluster_topk_patches import (
    TopPatch,
    build_image_to_shard,
    convert_state_to_top_patches,
    find_one,
    load_centroids,
    load_images,
    load_json,
    load_shards,
    open_patch_mmaps,
    parse_clusters,
    render_cluster,
    write_json,
    write_jsonl,
)


def mine_center_top_bottom_from_saved_distance(
    ids_path: Path,
    dist_path: Path,
    images: Sequence[Dict],
    clusters: Sequence[int],
    k: int,
    topk: int,
    metric: str,
    chunk_images: int,
    grid_w: int,
) -> Tuple[Dict[int, List[TopPatch]], Dict[int, List[TopPatch]]]:
    ids = np.load(ids_path, mmap_mode="r")
    dist = np.load(dist_path, mmap_mode="r")
    if ids.shape != dist.shape:
        raise ValueError(f"ids shape {ids.shape} != dist shape {dist.shape}")

    num_images, num_patches = ids.shape
    cluster_set = set(int(c) for c in clusters)
    use_all_clusters = len(cluster_set) == int(k)
    best_state: Dict[int, List[Tuple[float, int, int, float]]] = {int(c): [] for c in clusters}
    worst_state: Dict[int, List[Tuple[float, int, int, float]]] = {int(c): [] for c in clusters}

    for start in tqdm(range(0, num_images, chunk_images), desc="mine center top/bottom"):
        end = min(start + chunk_images, num_images)
        labels = np.asarray(ids[start:end]).reshape(-1)
        distance = np.asarray(dist[start:end]).astype("float32", copy=False).reshape(-1)
        scores = 1.0 - distance if metric == "cosine" else -distance
        flat = np.arange(labels.size, dtype=np.int64)

        if not use_all_clusters:
            mask = np.isin(labels, np.fromiter(cluster_set, dtype=labels.dtype))
            labels = labels[mask]
            distance = distance[mask]
            scores = scores[mask]
            flat = flat[mask]
        if labels.size == 0:
            continue

        order = np.lexsort((-scores, labels))
        labels = labels[order]
        distance = distance[order]
        scores = scores[order]
        flat = flat[order]

        uniq, first = np.unique(labels, return_index=True)
        bounds = list(first[1:]) + [len(labels)]
        for c, st, ed in zip(uniq.tolist(), first.tolist(), bounds):
            c = int(c)
            if c not in best_state:
                continue

            take = min(topk, ed - st)
            append_candidates(best_state[c], scores, distance, flat, start, num_patches, st, take, reverse=True, limit=topk)
            append_candidates(worst_state[c], scores, distance, flat, start, num_patches, ed - take, take, reverse=False, limit=topk)

    return (
        convert_state_to_top_patches(best_state, images, grid_w),
        convert_state_to_top_patches(worst_state, images, grid_w),
    )


def append_candidates(
    state: List[Tuple[float, int, int, float]],
    scores: np.ndarray,
    distance: np.ndarray,
    flat: np.ndarray,
    image_start: int,
    num_patches: int,
    offset: int,
    take: int,
    reverse: bool,
    limit: int,
) -> None:
    indices = range(offset, offset + take)
    if not reverse:
        indices = reversed(list(indices))

    for j in indices:
        local_flat = int(flat[j])
        image_index = int(image_start + local_flat // num_patches)
        patch_index = int(local_flat % num_patches)
        state.append((float(scores[j]), image_index, patch_index, float(distance[j])))

    state.sort(key=lambda x: x[0], reverse=reverse)
    del state[limit:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize closest and farthest assigned patch tokens for each cluster center.")
    parser.add_argument("--feature_dir", type=str, default="feature/coco2014_dinov2_vitb14_448")
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--clusters", type=str, default="all", help="all, one id like 17, comma list 1,2,3, or range 0-31")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--chunk_images", type=int, default=1024)
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--collage_cols", type=int, default=5)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--image_root", type=str, default=None, help="Optional fallback root for original COCO train2014 images.")
    parser.add_argument("--centroids", type=str, default=None)
    parser.add_argument("--cluster_ids", type=str, default=None)
    parser.add_argument("--cluster_dist", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--save_each", action="store_true", help="Save individual boxed/attention images in addition to collage.")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    image_root = Path(args.image_root) if args.image_root else None
    clusters = parse_clusters(args.clusters, args.k)

    meta = load_json(feature_dir / "meta.json") if (feature_dir / "meta.json").exists() else {}
    grid = meta.get("patch_grid", [32, 32])
    grid_h, grid_w = int(grid[0]), int(grid[1])

    centroids_path = Path(args.centroids) if args.centroids else feature_dir / "kmeans" / f"centroids_k{args.k}_fp32.npy"
    ids_path = Path(args.cluster_ids) if args.cluster_ids else find_one(f"patch_cluster_ids_k{args.k}_*.npy", feature_dir / "assignment")
    dist_path = Path(args.cluster_dist) if args.cluster_dist else feature_dir / "assignment" / f"patch_cluster_dist_k{args.k}_fp16.npy"
    if not dist_path.exists():
        raise FileNotFoundError(
            f"Cannot find distance file: {dist_path}. Re-run assignment with --save_dist, "
            "because this script mines closest/farthest assigned patches from saved distances."
        )

    out_dir = Path(args.out_dir) if args.out_dir else feature_dir / "vis" / f"center_top_bottom_patches_k{args.k}_top{args.topk}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] images: {feature_dir / 'images.jsonl'}")
    images = load_images(feature_dir, image_root=image_root)
    print(f"[load] centroids: {centroids_path}")
    centroids = load_centroids(centroids_path, args.metric)
    print(f"[mine] clusters={args.clusters}, topk={args.topk}, ids={ids_path}, dist={dist_path}")
    best_by_cluster, worst_by_cluster = mine_center_top_bottom_from_saved_distance(
        ids_path=ids_path,
        dist_path=dist_path,
        images=images,
        clusters=clusters,
        k=args.k,
        topk=args.topk,
        metric=args.metric,
        chunk_images=args.chunk_images,
        grid_w=grid_w,
    )

    summary_rows = []
    for c in clusters:
        for group, rows in (("most_similar", best_by_cluster.get(c, [])), ("least_similar", worst_by_cluster.get(c, []))):
            for item in rows:
                row = dict(item.__dict__)
                row["group"] = group
                summary_rows.append(row)
    write_jsonl(out_dir / "center_top_bottom_all_clusters.jsonl", summary_rows)

    print("[load] patch token shards for attention maps")
    shards = load_shards(feature_dir)
    image_to_shard = build_image_to_shard(shards)
    patch_mmaps = open_patch_mmaps(shards)

    config = {
        "feature_dir": str(feature_dir),
        "k": args.k,
        "topk": args.topk,
        "clusters": clusters,
        "metric": args.metric,
        "grid": [grid_h, grid_w],
        "centroids": str(centroids_path),
        "cluster_ids": str(ids_path),
        "cluster_dist": str(dist_path),
        "out_dir": str(out_dir),
        "groups": ["most_similar", "least_similar"],
        "note": "Patches are mined only from tokens assigned to each cluster. least_similar means lowest similarity / largest saved nearest distance within that cluster.",
    }
    write_json(out_dir / "config.json", config)

    for c in clusters:
        render_cluster(
            cluster_id=c,
            top_patches=best_by_cluster.get(c, []),
            centroids=centroids,
            image_to_shard=image_to_shard,
            patch_mmaps=patch_mmaps,
            out_dir=out_dir / "most_similar",
            metric=args.metric,
            grid_h=grid_h,
            grid_w=grid_w,
            tile_size=args.tile_size,
            collage_cols=args.collage_cols,
            alpha=args.overlay_alpha,
            save_each=args.save_each,
        )
        render_cluster(
            cluster_id=c,
            top_patches=worst_by_cluster.get(c, []),
            centroids=centroids,
            image_to_shard=image_to_shard,
            patch_mmaps=patch_mmaps,
            out_dir=out_dir / "least_similar",
            metric=args.metric,
            grid_h=grid_h,
            grid_w=grid_w,
            tile_size=args.tile_size,
            collage_cols=args.collage_cols,
            alpha=args.overlay_alpha,
            save_each=args.save_each,
        )

    print(f"[done] saved to {out_dir}")
    print(f"[save] {out_dir / 'center_top_bottom_all_clusters.jsonl'}")


if __name__ == "__main__":
    main()
