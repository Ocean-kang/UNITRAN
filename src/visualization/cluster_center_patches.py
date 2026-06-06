#!/usr/bin/env python3
"""
Visualize the most similar and least similar assigned patch tokens for each KMeans cluster center.

For each requested cluster, this script mines from saved assignment files:
  1. top-k unique images whose assigned patch token is closest to the cluster center;
  2. top-k unique images whose assigned patch token is farthest from the cluster center;
  3. cluster-level size and similarity/distance statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from src.visualization.cluster_patches import (
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

Candidate = Tuple[float, int, int, float]
Stats = Dict[int, Dict[str, float]]


def init_stats(clusters: Sequence[int]) -> Stats:
    return {
        int(c): {
            "cluster_id": int(c),
            "cluster_size": 0,
            "score_sum": 0.0,
            "score_min": float("inf"),
            "score_max": float("-inf"),
            "distance_sum": 0.0,
            "distance_min": float("inf"),
            "distance_max": float("-inf"),
        }
        for c in clusters
    }


def update_stats(stats: Stats, cluster_id: int, scores: np.ndarray, distance: np.ndarray) -> None:
    row = stats[int(cluster_id)]
    row["cluster_size"] += int(scores.size)
    row["score_sum"] += float(scores.sum(dtype=np.float64))
    row["score_min"] = min(float(row["score_min"]), float(scores.min()))
    row["score_max"] = max(float(row["score_max"]), float(scores.max()))
    row["distance_sum"] += float(distance.sum(dtype=np.float64))
    row["distance_min"] = min(float(row["distance_min"]), float(distance.min()))
    row["distance_max"] = max(float(row["distance_max"]), float(distance.max()))


def finalize_stats(stats: Stats) -> List[Dict[str, float]]:
    rows = []
    for cluster_id in sorted(stats):
        row = stats[cluster_id]
        size = int(row["cluster_size"])
        if size == 0:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": 0,
                    "similarity_min": None,
                    "similarity_mean": None,
                    "similarity_max": None,
                    "distance_min": None,
                    "distance_mean": None,
                    "distance_max": None,
                }
            )
            continue
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": size,
                "similarity_min": float(row["score_min"]),
                "similarity_mean": float(row["score_sum"] / size),
                "similarity_max": float(row["score_max"]),
                "distance_min": float(row["distance_min"]),
                "distance_mean": float(row["distance_sum"] / size),
                "distance_max": float(row["distance_max"]),
            }
        )
    return rows


def merge_unique_image_candidate(state: List[Candidate], candidate: Candidate, keep_high: bool, limit: int) -> None:
    score, image_index, patch_index, distance = candidate
    for i, old in enumerate(state):
        if old[1] != image_index:
            continue
        if (keep_high and score > old[0]) or ((not keep_high) and score < old[0]):
            state[i] = candidate
        break
    else:
        state.append(candidate)

    state.sort(key=lambda x: x[0], reverse=keep_high)
    del state[limit:]


def append_unique_image_candidates(
    state: List[Candidate],
    scores: np.ndarray,
    distance: np.ndarray,
    flat: np.ndarray,
    image_start: int,
    num_patches: int,
    start: int,
    end: int,
    keep_high: bool,
    limit: int,
) -> None:
    seen_images = set()
    indices = range(start, end) if keep_high else range(end - 1, start - 1, -1)
    for j in indices:
        local_flat = int(flat[j])
        image_index = int(image_start + local_flat // num_patches)
        if image_index in seen_images:
            continue
        seen_images.add(image_index)
        patch_index = int(local_flat % num_patches)
        merge_unique_image_candidate(
            state,
            (float(scores[j]), image_index, patch_index, float(distance[j])),
            keep_high=keep_high,
            limit=limit,
        )
        if len(seen_images) >= limit:
            break


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
) -> Tuple[Dict[int, List[TopPatch]], Dict[int, List[TopPatch]], List[Dict[str, float]]]:
    ids = np.load(ids_path, mmap_mode="r")
    dist = np.load(dist_path, mmap_mode="r")
    if ids.shape != dist.shape:
        raise ValueError(f"ids shape {ids.shape} != dist shape {dist.shape}")

    num_images, num_patches = ids.shape
    cluster_set = set(int(c) for c in clusters)
    use_all_clusters = len(cluster_set) == int(k)
    best_state: Dict[int, List[Candidate]] = {int(c): [] for c in clusters}
    worst_state: Dict[int, List[Candidate]] = {int(c): [] for c in clusters}
    stats = init_stats(clusters)

    for image_start in tqdm(range(0, num_images, chunk_images), desc="mine center top/bottom"):
        image_end = min(image_start + chunk_images, num_images)
        labels = np.asarray(ids[image_start:image_end]).reshape(-1)
        distance = np.asarray(dist[image_start:image_end]).astype("float32", copy=False).reshape(-1)
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
        for cluster_id, start, end in zip(uniq.tolist(), first.tolist(), bounds):
            cluster_id = int(cluster_id)
            if cluster_id not in best_state:
                continue

            update_stats(stats, cluster_id, scores[start:end], distance[start:end])
            append_unique_image_candidates(
                best_state[cluster_id], scores, distance, flat, image_start, num_patches, start, end, keep_high=True, limit=topk
            )
            append_unique_image_candidates(
                worst_state[cluster_id], scores, distance, flat, image_start, num_patches, start, end, keep_high=False, limit=topk
            )

    return (
        convert_state_to_top_patches(best_state, images, grid_w),
        convert_state_to_top_patches(worst_state, images, grid_w),
        finalize_stats(stats),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize closest and farthest assigned patch tokens for each cluster center.")
    parser.add_argument("--feature_dir", type=str, default="feature/coco2014_dinov2_vitb14_448")
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=20)
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
    parser.add_argument("--save_each", action="store_true", help="Save individual image/crop/attention images in addition to collages.")
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

    out_dir = Path(args.out_dir) if args.out_dir else feature_dir / "vis" / f"center_extremes_unique_k{args.k}_top{args.topk}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] images: {feature_dir / 'images.jsonl'}")
    images = load_images(feature_dir, image_root=image_root)
    print(f"[load] centroids: {centroids_path}")
    centroids = load_centroids(centroids_path, args.metric)
    print(f"[mine] clusters={args.clusters}, topk={args.topk}, ids={ids_path}, dist={dist_path}")
    best_by_cluster, worst_by_cluster, cluster_stats = mine_center_top_bottom_from_saved_distance(
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
    write_jsonl(out_dir / "center_extremes_unique_images.jsonl", summary_rows)
    write_jsonl(out_dir / "cluster_stats.jsonl", cluster_stats)

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
        "note": "Each group keeps at most one selected patch per image. Main collages are image+crop pairs; attention maps are saved as separate collage files.",
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
    print(f"[save] {out_dir / 'center_extremes_unique_images.jsonl'}")
    print(f"[save] {out_dir / 'cluster_stats.jsonl'}")


if __name__ == "__main__":
    main()
