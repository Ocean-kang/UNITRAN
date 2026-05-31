#!/usr/bin/env python3
"""
Visualize top-k patch tokens for each KMeans cluster.

This script uses files produced by `tools/extract_coco_dinov2_patch.py` and
`tools/cluster_coco_dinov2_streaming.py`:

feature/coco2014_dinov2_vitb14_448/
  images.jsonl
  patch_tokens/
    shard_000_fp16.mmap
    shard_000.meta.json
    ...
  kmeans/
    centroids_k1024_fp32.npy
  assignment/
    patch_cluster_ids_k1024_uint16.npy
    patch_cluster_dist_k1024_fp16.npy

For each requested cluster, it:
  1. finds top-k assigned patch tokens closest to the cluster centroid;
  2. maps each patch token back to its original image and 32x32 patch position;
  3. saves a marked original image with the selected patch boxed;
  4. computes a centroid-to-all-patches similarity map for that image;
  5. saves a collage for quick visual inspection.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm.auto import tqdm


@dataclass
class ShardInfo:
    shard_id: int
    mmap_path: Path
    shape: Tuple[int, int, int]
    dtype: str
    global_indices: np.ndarray
    global_to_local: Dict[int, int]


@dataclass
class TopPatch:
    cluster_id: int
    rank: int
    score: float
    distance: float
    image_index: int
    patch_index: int
    patch_row: int
    patch_col: int
    file_name: str
    image_path: str


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


def parse_clusters(text: str, k: int) -> List[int]:
    text = text.strip().lower()
    if text == "all":
        return list(range(k))
    clusters: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            clusters.extend(range(int(a), int(b) + 1))
        else:
            clusters.append(int(part))
    clusters = sorted(set(clusters))
    bad = [c for c in clusters if c < 0 or c >= k]
    if bad:
        raise ValueError(f"Cluster ids out of range [0,{k - 1}]: {bad[:10]}")
    return clusters


def find_one(pattern: str, root: Path, required: bool = True) -> Optional[Path]:
    matches = sorted(root.glob(pattern))
    if not matches:
        if required:
            raise FileNotFoundError(f"Cannot find {pattern} under {root}")
        return None
    if len(matches) > 1:
        print(f"[warn] multiple files match {pattern}; use {matches[0]}")
    return matches[0]


def load_images(feature_dir: Path, image_root: Optional[Path]) -> List[Dict]:
    rows = load_jsonl(feature_dir / "images.jsonl")
    rows = sorted(rows, key=lambda r: int(r.get("index", len(rows))))
    for row in rows:
        path = Path(str(row.get("path", "")))
        if path.exists():
            row["resolved_path"] = str(path)
            continue
        if image_root is not None:
            candidate = image_root / row.get("file_name", "")
            if candidate.exists():
                row["resolved_path"] = str(candidate)
                continue
        row["resolved_path"] = str(path)
    return rows


def load_shards(feature_dir: Path) -> List[ShardInfo]:
    patch_dir = feature_dir / "patch_tokens"
    shards = []
    for meta_path in sorted(patch_dir.glob("shard_*.meta.json")):
        meta = load_json(meta_path)
        shape = tuple(int(x) for x in meta["shape"])
        dtype = str(meta.get("dtype", "float16"))
        global_indices = np.asarray(meta.get("global_indices", list(range(shape[0]))), dtype=np.int64)
        mmap_value = str(meta.get("mmap_path", ""))
        mmap_path = Path(mmap_value)
        if not mmap_path.is_absolute():
            candidate = feature_dir / mmap_value
            mmap_path = candidate if candidate.exists() else patch_dir / Path(mmap_value).name
        if not mmap_path.exists():
            suffix = "fp16" if dtype == "float16" else "fp32"
            mmap_path = patch_dir / f"shard_{int(meta.get('shard_id', len(shards))):03d}_{suffix}.mmap"
        if not mmap_path.exists():
            raise FileNotFoundError(f"Cannot locate mmap for {meta_path}: {mmap_path}")
        global_to_local = {int(g): i for i, g in enumerate(global_indices.tolist())}
        shards.append(
            ShardInfo(
                shard_id=int(meta.get("shard_id", len(shards))),
                mmap_path=mmap_path,
                shape=shape,
                dtype=dtype,
                global_indices=global_indices,
                global_to_local=global_to_local,
            )
        )
    if not shards:
        raise FileNotFoundError(f"No shard_*.meta.json found under {patch_dir}")
    return shards


def build_image_to_shard(shards: Sequence[ShardInfo]) -> Dict[int, Tuple[ShardInfo, int]]:
    out: Dict[int, Tuple[ShardInfo, int]] = {}
    for shard in shards:
        for global_i, local_i in shard.global_to_local.items():
            out[int(global_i)] = (shard, int(local_i))
    return out


def open_patch_mmaps(shards: Sequence[ShardInfo]) -> Dict[int, np.memmap]:
    mmaps = {}
    for shard in shards:
        dtype = np.float16 if shard.dtype == "float16" else np.float32
        mmaps[shard.shard_id] = np.memmap(shard.mmap_path, dtype=dtype, mode="r", shape=shard.shape)
    return mmaps


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype("float32", copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def load_centroids(path: Path, metric: str) -> np.ndarray:
    centroids = np.load(path).astype("float32", copy=False)
    if metric == "cosine":
        centroids = normalize_rows(centroids)
    return centroids


def mine_topk_from_saved_distance(
    ids_path: Path,
    dist_path: Path,
    images: Sequence[Dict],
    clusters: Sequence[int],
    topk: int,
    metric: str,
    chunk_images: int,
    grid_w: int,
) -> Dict[int, List[TopPatch]]:
    ids = np.load(ids_path, mmap_mode="r")
    dist = np.load(dist_path, mmap_mode="r")
    if ids.shape != dist.shape:
        raise ValueError(f"ids shape {ids.shape} != dist shape {dist.shape}")
    num_images, num_patches = ids.shape
    cluster_set = set(int(c) for c in clusters)
    all_clusters = len(cluster_set) == 0 or len(cluster_set) >= int(ids.max()) + 1
    state: Dict[int, List[Tuple[float, int, int, float]]] = {int(c): [] for c in clusters}

    for start in tqdm(range(0, num_images, chunk_images), desc="mine topk"):
        end = min(start + chunk_images, num_images)
        labels = np.asarray(ids[start:end]).reshape(-1)
        distance = np.asarray(dist[start:end]).astype("float32", copy=False).reshape(-1)
        scores = 1.0 - distance if metric == "cosine" else -distance
        flat = np.arange(labels.size, dtype=np.int64)

        if not all_clusters:
            mask = np.isin(labels, np.fromiter(cluster_set, dtype=labels.dtype))
            labels = labels[mask]
            scores = scores[mask]
            distance = distance[mask]
            flat = flat[mask]
        if labels.size == 0:
            continue

        order = np.lexsort((-scores, labels))
        labels = labels[order]
        scores = scores[order]
        distance = distance[order]
        flat = flat[order]

        uniq, first = np.unique(labels, return_index=True)
        bounds = list(first[1:]) + [len(labels)]
        for c, st, ed in zip(uniq.tolist(), first.tolist(), bounds):
            c = int(c)
            if c not in state:
                continue
            take = min(topk, ed - st)
            local_flat = flat[st: st + take]
            local_img = local_flat // num_patches
            patch_idx = local_flat % num_patches
            candidates = state[c]
            for j in range(take):
                img_i = int(start + local_img[j])
                candidates.append((float(scores[st + j]), img_i, int(patch_idx[j]), float(distance[st + j])))
            candidates.sort(key=lambda x: x[0], reverse=True)
            del candidates[topk:]

    return convert_state_to_top_patches(state, images, grid_w)


def convert_state_to_top_patches(
    state: Dict[int, List[Tuple[float, int, int, float]]],
    images: Sequence[Dict],
    grid_w: int,
) -> Dict[int, List[TopPatch]]:
    out: Dict[int, List[TopPatch]] = {}
    for c, items in state.items():
        rows: List[TopPatch] = []
        for rank, (score, image_index, patch_index, distance) in enumerate(items, start=1):
            img = images[image_index]
            rows.append(
                TopPatch(
                    cluster_id=int(c),
                    rank=rank,
                    score=float(score),
                    distance=float(distance),
                    image_index=int(image_index),
                    patch_index=int(patch_index),
                    patch_row=int(patch_index // grid_w),
                    patch_col=int(patch_index % grid_w),
                    file_name=str(img.get("file_name", "")),
                    image_path=str(img.get("resolved_path", img.get("path", ""))),
                )
            )
        out[int(c)] = rows
    return out


def patch_box(width: int, height: int, row: int, col: int, grid_h: int, grid_w: int) -> Tuple[int, int, int, int]:
    x0 = int(round(col * width / grid_w))
    y0 = int(round(row * height / grid_h))
    x1 = int(round((col + 1) * width / grid_w))
    y1 = int(round((row + 1) * height / grid_h))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def draw_patch_box(im: Image.Image, row: int, col: int, grid_h: int, grid_w: int, width: int = 5) -> Image.Image:
    im = im.convert("RGB")
    draw = ImageDraw.Draw(im)
    box = patch_box(im.width, im.height, row, col, grid_h, grid_w)
    for i in range(width):
        draw.rectangle((box[0] - i, box[1] - i, box[2] + i, box[3] + i), outline=(255, 0, 0))
    return im


def simple_colormap(values: np.ndarray) -> np.ndarray:
    """Return an RGB heatmap without requiring matplotlib."""
    v = np.clip(values.astype("float32"), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def centroid_attention_map(tokens: np.ndarray, centroid: np.ndarray, metric: str) -> np.ndarray:
    x = tokens.astype("float32", copy=False)
    c = centroid.astype("float32", copy=False)
    if metric == "cosine":
        x = normalize_rows(x)
        c = c / max(float(np.linalg.norm(c)), 1e-12)
        score = x @ c
    else:
        diff = x - c[None, :]
        score = -np.sum(diff * diff, axis=1)
    score = score.astype("float32", copy=False)
    score = (score - score.min()) / max(float(score.max() - score.min()), 1e-12)
    return score


def make_attention_overlay(im: Image.Image, attention: np.ndarray, alpha: float) -> Image.Image:
    heat = simple_colormap(attention)
    heat = (heat * 255.0).astype("uint8")
    heat_im = Image.fromarray(heat, mode="RGB").resize(im.size, resample=Image.Resampling.BICUBIC)
    return Image.blend(im.convert("RGB"), heat_im, alpha=alpha)


def fit_square(im: Image.Image, size: int) -> Image.Image:
    im = ImageOps.contain(im.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    x = (size - im.width) // 2
    y = (size - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def add_label(im: Image.Image, text: str, label_h: int = 24) -> Image.Image:
    font = ImageFont.load_default()
    canvas = Image.new("RGB", (im.width, im.height + label_h), (255, 255, 255))
    canvas.paste(im, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), text, fill=(0, 0, 0), font=font)
    return canvas


def save_collage(tiles: Sequence[Image.Image], path: Path, cols: int) -> None:
    if not tiles:
        return
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    rows = int(math.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, tile in enumerate(tiles):
        x = (i % cols) * w
        y = (i // cols) * h
        canvas.paste(tile, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def render_cluster(
    cluster_id: int,
    top_patches: Sequence[TopPatch],
    centroids: np.ndarray,
    image_to_shard: Dict[int, Tuple[ShardInfo, int]],
    patch_mmaps: Dict[int, np.memmap],
    out_dir: Path,
    metric: str,
    grid_h: int,
    grid_w: int,
    tile_size: int,
    collage_cols: int,
    alpha: float,
    save_each: bool,
) -> None:
    cluster_dir = out_dir / f"cluster_{cluster_id:04d}"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tiles = []
    centroid = centroids[cluster_id]

    for item in tqdm(top_patches, desc=f"render cluster {cluster_id:04d}", leave=False):
        img_path = Path(item.image_path)
        if not img_path.exists():
            print(f"[warn] image not found: {img_path}")
            continue
        im = Image.open(img_path).convert("RGB")
        boxed = draw_patch_box(im.copy(), item.patch_row, item.patch_col, grid_h, grid_w)

        if item.image_index not in image_to_shard:
            print(f"[warn] image_index {item.image_index} not found in patch token shards")
            continue
        shard, local_i = image_to_shard[item.image_index]
        tokens = np.asarray(patch_mmaps[shard.shard_id][local_i])
        attention = centroid_attention_map(tokens, centroid, metric).reshape(grid_h, grid_w)
        overlay = make_attention_overlay(im, attention, alpha=alpha)
        overlay = draw_patch_box(overlay, item.patch_row, item.patch_col, grid_h, grid_w, width=4)

        label = f"r{item.rank:03d} img{item.image_index} p{item.patch_index} s{item.score:.4f}"
        pair = Image.new("RGB", (tile_size * 2, tile_size + 24), (255, 255, 255))
        pair.paste(add_label(fit_square(boxed, tile_size), "box | " + label), (0, 0))
        pair.paste(add_label(fit_square(overlay, tile_size), "map | " + label), (tile_size, 0))
        tiles.append(pair)

        if save_each:
            stem = f"rank_{item.rank:03d}_img_{item.image_index:06d}_patch_{item.patch_index:04d}_score_{item.score:.4f}"
            boxed.save(cluster_dir / f"{stem}_box.jpg", quality=95)
            overlay.save(cluster_dir / f"{stem}_attn.jpg", quality=95)

        rows.append(item.__dict__)

    write_jsonl(cluster_dir / "topk_meta.jsonl", rows)
    save_collage(tiles, cluster_dir / f"cluster_{cluster_id:04d}_top{len(tiles)}_collage.jpg", cols=collage_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize top-k patch tokens for each DINOv2 patch cluster.")
    parser.add_argument("--feature_dir", type=str, default="feature/coco2014_dinov2_vitb14_448")
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=50)
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
            "or extend this script to recompute distances from patch tokens."
        )

    out_dir = Path(args.out_dir) if args.out_dir else feature_dir / "vis" / f"topk_cluster_patches_k{args.k}_top{args.topk}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] images: {feature_dir / 'images.jsonl'}")
    images = load_images(feature_dir, image_root=image_root)
    print(f"[load] centroids: {centroids_path}")
    centroids = load_centroids(centroids_path, args.metric)
    print(f"[mine] clusters={args.clusters}, topk={args.topk}, ids={ids_path}, dist={dist_path}")
    topk_by_cluster = mine_topk_from_saved_distance(
        ids_path=ids_path,
        dist_path=dist_path,
        images=images,
        clusters=clusters,
        topk=args.topk,
        metric=args.metric,
        chunk_images=args.chunk_images,
        grid_w=grid_w,
    )

    summary_rows = []
    for c in clusters:
        for item in topk_by_cluster.get(c, []):
            summary_rows.append(item.__dict__)
    write_jsonl(out_dir / "topk_all_clusters.jsonl", summary_rows)

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
        "note": "attention map means centroid-to-all-patches similarity map, not transformer self-attention.",
    }
    write_json(out_dir / "config.json", config)

    for c in clusters:
        render_cluster(
            cluster_id=c,
            top_patches=topk_by_cluster.get(c, []),
            centroids=centroids,
            image_to_shard=image_to_shard,
            patch_mmaps=patch_mmaps,
            out_dir=out_dir,
            metric=args.metric,
            grid_h=grid_h,
            grid_w=grid_w,
            tile_size=args.tile_size,
            collage_cols=args.collage_cols,
            alpha=args.overlay_alpha,
            save_each=args.save_each,
        )

    print(f"[done] saved to {out_dir}")
    print(f"[save] {out_dir / 'topk_all_clusters.jsonl'}")


if __name__ == "__main__":
    main()
