#!/usr/bin/env python3
"""Visualize one image's per-image local KMeans centers over raw DINOv2 patch tokens."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.features.patch_store import find_image_row, load_json, write_json, write_jsonl
from src.visualization.cluster_patches import (
    add_label,
    build_image_to_shard,
    centroid_attention_map,
    draw_patch_box,
    fit_square,
    load_images,
    load_shards,
    make_attention_overlay,
    open_patch_mmaps,
    parse_clusters,
    save_collage,
    simple_colormap,
)


def resolve_raw_feature_dir(local_feature_dir: Path, raw_feature_dir: Optional[str]) -> Path:
    if raw_feature_dir:
        return Path(raw_feature_dir)
    meta_path = local_feature_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Cannot infer raw feature dir because {meta_path} does not exist. Use --raw_feature_dir.")
    meta = load_json(meta_path)
    source = meta.get("source_feature_dir")
    if not source:
        raise ValueError(f"Cannot infer raw feature dir because source_feature_dir is missing in {meta_path}. Use --raw_feature_dir.")
    return Path(str(source))


def get_image_tokens(feature_dir: Path, image_index: int) -> np.ndarray:
    shards = load_shards(feature_dir)
    image_to_shard = build_image_to_shard(shards)
    if image_index not in image_to_shard:
        raise KeyError(f"image_index={image_index} is not found in {feature_dir}/patch_tokens")
    mmaps = open_patch_mmaps(shards)
    shard, local_i = image_to_shard[image_index]
    return np.asarray(mmaps[shard.shard_id][local_i]).astype("float32", copy=False)


def save_heatmap_image(attention: np.ndarray, image_size: Sequence[int], path: Path) -> None:
    heat = (simple_colormap(attention) * 255.0).astype("uint8")
    heat_im = Image.fromarray(heat, mode="RGB")
    resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    heat_im = heat_im.resize((int(image_size[0]), int(image_size[1])), resample=resample)
    path.parent.mkdir(parents=True, exist_ok=True)
    heat_im.save(path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one image's local-center-to-raw-patch attention maps.")
    parser.add_argument("--local_feature_dir", type=str, default="feature/coco2014_dinov2_vitb14_448_local_k20")
    parser.add_argument("--raw_feature_dir", type=str, default=None, help="Default: read source_feature_dir from local_feature_dir/meta.json.")
    parser.add_argument("--image_name", type=str, default=None, help="Example: COCO_train2014_000000000009.jpg")
    parser.add_argument("--image_index", type=int, default=None, help="Optional exact image index from images.jsonl.")
    parser.add_argument("--local_centers", type=str, default="all", help="all, one id like 7, comma list 1,2,3, or range 0-19")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--image_root", type=str, default=None, help="Optional fallback root for original COCO train2014 images.")
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--collage_cols", type=int, default=5)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    local_feature_dir = Path(args.local_feature_dir)
    raw_feature_dir = resolve_raw_feature_dir(local_feature_dir, args.raw_feature_dir)
    image_root = Path(args.image_root) if args.image_root else None

    raw_meta = load_json(raw_feature_dir / "meta.json") if (raw_feature_dir / "meta.json").exists() else {}
    grid = raw_meta.get("patch_grid", [32, 32])
    grid_h, grid_w = int(grid[0]), int(grid[1])

    images = load_images(raw_feature_dir, image_root=image_root)
    image_row = find_image_row(images, args.image_name, args.image_index)
    image_index = int(image_row["index"])
    image_path = Path(str(image_row.get("resolved_path", image_row.get("path", ""))))
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot find image file: {image_path}. Use --image_root if paths moved.")

    raw_tokens = get_image_tokens(raw_feature_dir, image_index)
    if raw_tokens.shape[0] != grid_h * grid_w:
        raise ValueError(f"raw token count {raw_tokens.shape[0]} != grid size {grid_h}x{grid_w}")

    local_centers = get_image_tokens(local_feature_dir, image_index)
    local_k = int(local_centers.shape[0])
    selected_centers = parse_clusters(args.local_centers, local_k)

    stem = image_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else local_feature_dir / "vis" / "image_local_centers_attention" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    tiles = []
    with Image.open(image_path) as im0:
        im = im0.convert("RGB")

    for center_id in tqdm(selected_centers, desc="render local centers"):
        center = local_centers[center_id]
        attention = centroid_attention_map(raw_tokens, center, args.metric).reshape(grid_h, grid_w)
        flat_argmax = int(attention.reshape(-1).argmax())
        patch_row = int(flat_argmax // grid_w)
        patch_col = int(flat_argmax % grid_w)

        center_dir = out_dir / f"local_center_{center_id:03d}"
        center_dir.mkdir(parents=True, exist_ok=True)
        npy_path = center_dir / "attention.npy"
        heat_path = center_dir / "attention_heatmap.jpg"
        overlay_path = center_dir / "attention_overlay.jpg"
        np.save(npy_path, attention)
        save_heatmap_image(attention, im.size, heat_path)
        overlay = make_attention_overlay(im, attention, alpha=args.overlay_alpha)
        overlay = draw_patch_box(overlay, patch_row, patch_col, grid_h, grid_w, width=4)
        overlay.save(overlay_path, quality=95)

        label = f"center{center_id:03d} max_p{flat_argmax:04d} v{attention.max():.3f}"
        tiles.append(add_label(fit_square(overlay, args.tile_size), label))
        rows.append(
            {
                "image_index": image_index,
                "file_name": image_row.get("file_name"),
                "local_center_index": int(center_id),
                "max_patch_index": flat_argmax,
                "max_patch_row": patch_row,
                "max_patch_col": patch_col,
                "max_normalized_attention": float(attention.max()),
                "mean_normalized_attention": float(attention.mean()),
                "attention_npy": str(npy_path),
                "heatmap": str(heat_path),
                "overlay": str(overlay_path),
            }
        )

    collage_path = out_dir / f"{stem}_local_centers_{args.local_centers}_collage.jpg"
    save_collage(tiles, collage_path, cols=args.collage_cols)
    rows_path = out_dir / "local_center_attention_meta.jsonl"
    write_jsonl(rows_path, rows)

    config = {
        "raw_feature_dir": str(raw_feature_dir),
        "local_feature_dir": str(local_feature_dir),
        "image": {
            "index": image_index,
            "image_id": image_row.get("image_id"),
            "file_name": image_row.get("file_name"),
            "path": str(image_path),
        },
        "metric": args.metric,
        "grid": [grid_h, grid_w],
        "local_k": local_k,
        "local_centers": selected_centers,
        "attention_source": "local_center_to_raw_patch_token_similarity",
        "outputs": {
            "collage": str(collage_path),
            "meta_jsonl": str(rows_path),
        },
    }
    write_json(out_dir / "config.json", config)

    print(f"[image] index={image_index}, file_name={image_row.get('file_name')}")
    print(f"[save] {collage_path}")
    print(f"[save] {rows_path}")
    print(f"[save] {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
