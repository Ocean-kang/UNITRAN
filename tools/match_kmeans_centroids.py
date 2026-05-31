#!/usr/bin/env python3
"""Match high-K KMeans centroids to low-K centroids by cosine similarity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype("float32", copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def load_centroids(path: Path, expected_k: int) -> np.ndarray:
    centroids = np.load(path).astype("float32", copy=False)
    if centroids.ndim != 2:
        raise ValueError(f"Expected [K,D] centroid array, got {centroids.shape}: {path}")
    if centroids.shape[0] != expected_k:
        raise ValueError(f"Expected K={expected_k}, got {centroids.shape[0]}: {path}")
    return np.ascontiguousarray(normalize_rows(centroids))


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "small_index",
            "large_index",
            "cosine_similarity",
            "rank_in_small_to_large",
            "best_large_index_for_small",
            "best_cosine_for_small",
            "gap_to_best",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def build_match_rows(sim: np.ndarray, matched_large: np.ndarray) -> List[Dict]:
    best_large = np.argmax(sim, axis=1)
    best_score = sim[np.arange(sim.shape[0]), best_large]

    rows: List[Dict] = []
    for small_i, large_i in enumerate(matched_large.tolist()):
        score = float(sim[small_i, large_i])
        rows.append(
            {
                "small_index": int(small_i),
                "large_index": int(large_i),
                "cosine_similarity": score,
                "rank_in_small_to_large": int(np.count_nonzero(sim[small_i] > score) + 1),
                "best_large_index_for_small": int(best_large[small_i]),
                "best_cosine_for_small": float(best_score[small_i]),
                "gap_to_best": float(best_score[small_i] - score),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Match high-K KMeans centroids to low-K centroids by cosine similarity.")
    parser.add_argument("--feature_dir", type=str, default="feature/coco2014_dinov2_vitb14_448")
    parser.add_argument("--small_k", type=int, default=512)
    parser.add_argument("--large_k", type=int, default=4096)
    parser.add_argument("--small_centroids", type=str, default=None)
    parser.add_argument("--large_centroids", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.large_k < args.small_k:
        raise ValueError(f"large_k must be >= small_k, got {args.large_k} < {args.small_k}")

    feature_dir = Path(args.feature_dir)
    kmeans_dir = feature_dir / "kmeans"
    small_path = Path(args.small_centroids) if args.small_centroids else kmeans_dir / f"centroids_k{args.small_k}_fp32.npy"
    large_path = Path(args.large_centroids) if args.large_centroids else kmeans_dir / f"centroids_k{args.large_k}_fp32.npy"
    out_dir = Path(args.out_dir) if args.out_dir else feature_dir / "centroid_match" / f"k{args.large_k}_to_k{args.small_k}"

    matches_jsonl = out_dir / "matches.jsonl"
    if matches_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"{matches_jsonl} exists. Use --overwrite to rewrite outputs.")

    print(f"[load] small centroids: {small_path}")
    small = load_centroids(small_path, args.small_k)
    print(f"[load] large centroids: {large_path}")
    large = load_centroids(large_path, args.large_k)
    if small.shape[1] != large.shape[1]:
        raise ValueError(f"Centroid dim mismatch: small={small.shape}, large={large.shape}")

    print(f"[match] cosine similarity matrix: ({args.small_k}, {args.large_k})")
    sim = small @ large.T

    small_indices, large_indices = linear_sum_assignment(-sim)
    order = np.argsort(small_indices)
    matched_large = large_indices[order].astype(np.int64)
    rows = build_match_rows(sim, matched_large)

    pair_indices = np.stack([np.arange(args.small_k, dtype=np.int64), matched_large], axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(matches_jsonl, rows)
    write_csv(out_dir / "matches.csv", rows)
    np.save(out_dir / "matched_pair_indices.npy", pair_indices)
    np.save(out_dir / "matched_large_indices.npy", matched_large)

    config = {
        "feature_dir": str(feature_dir),
        "small_k": args.small_k,
        "large_k": args.large_k,
        "small_centroids": str(small_path),
        "large_centroids": str(large_path),
        "out_dir": str(out_dir),
        "method": "hungarian_one_to_one_max_total_cosine",
        "num_pairs": int(args.small_k),
        "unique_large_indices": int(len(set(matched_large.tolist()))),
        "mean_cosine_similarity": float(np.mean([r["cosine_similarity"] for r in rows])),
        "min_cosine_similarity": float(np.min([r["cosine_similarity"] for r in rows])),
        "note": "Each row maps one small_index to one unique large_index. rank_in_small_to_large=1 means the selected high-K centroid is also that low-K centroid's nearest high-K centroid.",
    }
    write_json(out_dir / "config.json", config)

    print(f"[done] pairs={args.small_k}, unique_large={config['unique_large_indices']}, mean_cos={config['mean_cosine_similarity']:.6f}")
    print(f"[save] {matches_jsonl}")
    print(f"[save] {out_dir / 'matches.csv'}")
    print(f"[save] {out_dir / 'matched_pair_indices.npy'}")
    print(f"[save] {out_dir / 'matched_large_indices.npy'}")
    print(f"[save] {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
