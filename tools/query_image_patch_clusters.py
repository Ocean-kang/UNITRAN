#!/usr/bin/env python3
"""Query which KMeans cluster each patch token of one image belongs to."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_grid_csv(path: Path, grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(grid.tolist())
    tmp.replace(path)


def parse_k_values(text: str, assignment_dir: Path) -> List[int]:
    text = text.strip().lower()
    if text == "all":
        ks = []
        for path in assignment_dir.glob("patch_cluster_ids_k*.npy"):
            match = re.search(r"patch_cluster_ids_k(\d+)_", path.name)
            if match:
                ks.append(int(match.group(1)))
        if not ks:
            raise FileNotFoundError(f"No patch_cluster_ids_k*.npy found under {assignment_dir}")
        return sorted(set(ks))

    ks = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not ks:
        raise ValueError("--ks is empty")
    return sorted(set(ks))


def find_one(pattern: str, root: Path, required: bool = True) -> Optional[Path]:
    matches = sorted(root.glob(pattern))
    if not matches:
        if required:
            raise FileNotFoundError(f"Cannot find {pattern} under {root}")
        return None
    if len(matches) > 1:
        print(f"[warn] multiple files match {pattern}; use {matches[0]}")
    return matches[0]


def load_images(feature_dir: Path) -> List[Dict]:
    rows = load_jsonl(feature_dir / "images.jsonl")
    return sorted(rows, key=lambda r: int(r.get("index", len(rows))))


def find_image_row(images: Sequence[Dict], image_name: Optional[str], image_index: Optional[int]) -> Dict:
    if image_index is not None:
        for row in images:
            if int(row.get("index", -1)) == image_index:
                return row
        raise KeyError(f"No image with index={image_index}")

    if image_name is None:
        raise ValueError("Use --image_name or --image_index")

    target = Path(image_name).name
    matches = [row for row in images if Path(str(row.get("file_name", ""))).name == target]
    if not matches:
        raise KeyError(f"No image named {target} in images.jsonl")
    if len(matches) > 1:
        raise ValueError(f"Multiple images named {target}; use --image_index instead")
    return matches[0]


def load_distance_for_k(assignment_dir: Path, k: int) -> Optional[np.ndarray]:
    path = assignment_dir / f"patch_cluster_dist_k{k}_fp16.npy"
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def build_patch_rows(cluster_ids: np.ndarray, distances: Optional[np.ndarray], grid_w: int) -> List[Dict]:
    rows = []
    flat_ids = cluster_ids.reshape(-1)
    flat_dist = distances.reshape(-1) if distances is not None else None
    for patch_index, cluster_id in enumerate(flat_ids.tolist()):
        row = {
            "patch_index": int(patch_index),
            "patch_row": int(patch_index // grid_w),
            "patch_col": int(patch_index % grid_w),
            "cluster_id": int(cluster_id),
        }
        if flat_dist is not None:
            distance = float(flat_dist[patch_index])
            row["distance"] = distance
            row["cosine_similarity"] = float(1.0 - distance)
        rows.append(row)
    return rows


def summarize_clusters(cluster_ids: np.ndarray, distances: Optional[np.ndarray]) -> List[Dict]:
    flat_ids = cluster_ids.reshape(-1).astype(np.int64, copy=False)
    uniq, counts = np.unique(flat_ids, return_counts=True)
    order = np.argsort(-counts)
    total = int(flat_ids.size)
    summary = []
    flat_dist = distances.reshape(-1).astype("float32", copy=False) if distances is not None else None

    for idx in order.tolist():
        cluster_id = int(uniq[idx])
        count = int(counts[idx])
        item = {
            "cluster_id": cluster_id,
            "patch_count": count,
            "patch_ratio": float(count / total),
        }
        if flat_dist is not None:
            mask = flat_ids == cluster_id
            d = flat_dist[mask]
            item["mean_distance"] = float(d.mean())
            item["min_distance"] = float(d.min())
            item["max_distance"] = float(d.max())
            item["mean_cosine_similarity"] = float(1.0 - d.mean())
        summary.append(item)
    return summary


def process_one_k(
    feature_dir: Path,
    out_root: Path,
    image_row: Dict,
    k: int,
    grid_h: int,
    grid_w: int,
) -> Dict:
    assignment_dir = feature_dir / "assignment"
    ids_path = find_one(f"patch_cluster_ids_k{k}_*.npy", assignment_dir)
    assert ids_path is not None
    ids = np.load(ids_path, mmap_mode="r")

    image_index = int(image_row["index"])
    if image_index < 0 or image_index >= ids.shape[0]:
        raise IndexError(f"image index {image_index} is outside assignment shape {ids.shape}")
    if ids.shape[1] != grid_h * grid_w:
        raise ValueError(f"assignment patch count {ids.shape[1]} != grid size {grid_h}x{grid_w}")

    cluster_grid = np.asarray(ids[image_index]).reshape(grid_h, grid_w)
    dist_all = load_distance_for_k(assignment_dir, k)
    distance_grid = None
    if dist_all is not None:
        if dist_all.shape != ids.shape:
            raise ValueError(f"distance shape {dist_all.shape} != assignment shape {ids.shape}")
        distance_grid = np.asarray(dist_all[image_index]).astype("float32", copy=False).reshape(grid_h, grid_w)

    out_dir = out_root / f"k{k}"
    npy_path = out_dir / f"patch_clusters_grid_k{k}.npy"
    csv_path = out_dir / f"patch_clusters_grid_k{k}.csv"
    patch_rows_path = out_dir / f"patch_clusters_flat_k{k}.jsonl"
    summary_path = out_dir / f"patch_clusters_summary_k{k}.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, cluster_grid)
    write_grid_csv(csv_path, cluster_grid)
    write_jsonl(patch_rows_path, build_patch_rows(cluster_grid, distance_grid, grid_w))

    cluster_summary = summarize_clusters(cluster_grid, distance_grid)
    payload = {
        "feature_dir": str(feature_dir),
        "k": int(k),
        "image": {
            "index": image_index,
            "image_id": image_row.get("image_id"),
            "file_name": image_row.get("file_name"),
            "path": image_row.get("path"),
        },
        "grid": [int(grid_h), int(grid_w)],
        "num_patches": int(grid_h * grid_w),
        "num_unique_clusters": int(len(cluster_summary)),
        "cluster_ids_path": str(ids_path),
        "cluster_dist_path": str(assignment_dir / f"patch_cluster_dist_k{k}_fp16.npy") if distance_grid is not None else None,
        "outputs": {
            "grid_npy": str(npy_path),
            "grid_csv": str(csv_path),
            "flat_jsonl": str(patch_rows_path),
            "summary_json": str(summary_path),
        },
        "clusters_by_patch_count": cluster_summary,
    }
    write_json(summary_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Query one image's patch-token KMeans cluster ids.")
    parser.add_argument("--feature_dir", type=str, default="feature/coco2014_dinov2_vitb14_448")
    parser.add_argument("--image_name", type=str, default=None, help="Example: COCO_train2014_000000000009.jpg")
    parser.add_argument("--image_index", type=int, default=None, help="Optional exact image index from images.jsonl.")
    parser.add_argument("--ks", type=str, default="all", help="all, one K like 512, or comma list like 512,1024,2048.")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    images = load_images(feature_dir)
    image_row = find_image_row(images, args.image_name, args.image_index)

    meta = load_json(feature_dir / "meta.json") if (feature_dir / "meta.json").exists() else {}
    grid = meta.get("patch_grid", [32, 32])
    grid_h, grid_w = int(grid[0]), int(grid[1])

    ks = parse_k_values(args.ks, feature_dir / "assignment")
    stem = Path(str(image_row.get("file_name", image_row["index"]))).stem
    out_root = Path(args.out_dir) if args.out_dir else feature_dir / "query_image_patch_clusters" / stem
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[image] index={image_row['index']}, file_name={image_row.get('file_name')}")
    print(f"[query] ks={ks}, grid={grid_h}x{grid_w}")

    all_summaries = []
    for k in ks:
        result = process_one_k(
            feature_dir=feature_dir,
            out_root=out_root,
            image_row=image_row,
            k=k,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        top = result["clusters_by_patch_count"][:10]
        all_summaries.append(
            {
                "k": int(k),
                "num_unique_clusters": result["num_unique_clusters"],
                "top_clusters_by_patch_count": top,
                "summary_json": result["outputs"]["summary_json"],
            }
        )
        print(f"[k={k}] unique_clusters={result['num_unique_clusters']}, top10={[(x['cluster_id'], x['patch_count']) for x in top]}")
        print(f"[save] {result['outputs']['summary_json']}")

    index_path = out_root / "index.json"
    write_json(
        index_path,
        {
            "feature_dir": str(feature_dir),
            "image": {
                "index": int(image_row["index"]),
                "image_id": image_row.get("image_id"),
                "file_name": image_row.get("file_name"),
                "path": image_row.get("path"),
            },
            "grid": [grid_h, grid_w],
            "ks": ks,
            "summaries": all_summaries,
        },
    )
    print(f"[done] saved to {out_root}")
    print(f"[save] {index_path}")


if __name__ == "__main__":
    main()
