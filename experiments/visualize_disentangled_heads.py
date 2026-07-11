#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize disentangled_self_attn heads as heatmap / overlay grids.

Input:
  feature_dir/disentangled_self_attn/shard_000_fp16.mmap  # [N, H, D]
  feature_dir/patch_tokens/shard_000_fp16.mmap            # [N, P, D]
  feature_dir/images.jsonl or patch_tokens/shard_000.images.jsonl

Important:
  disentangled_self_attn is [N, H, D], without spatial dimension.
  This script computes spatial response maps by:
    score[h, p] = cosine(disentangled_head[h], patch_token[p])

Output:
  output_dir/heatmap_grid/*.png
  output_dir/overlay_grid/*.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import matplotlib.cm as mpl_cm
except Exception:
    mpl_cm = None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_indices(indices: Optional[str], n: int, max_images: Optional[int]) -> List[int]:
    if indices:
        out = []
        for item in indices.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                a, b = item.split("-", 1)
                a, b = int(a), int(b)
                out.extend(list(range(a, b + 1)))
            else:
                out.append(int(item))
        out = [i for i in out if 0 <= i < n]
        if max_images is not None:
            out = out[:max_images]
        return out

    out = list(range(n))
    if max_images is not None:
        out = out[:max_images]
    return out


def read_shape_from_json(obj: Any) -> Optional[Tuple[int, ...]]:
    if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
        return tuple(obj)

    if isinstance(obj, dict):
        for key in ["shape", "mmap_shape", "array_shape"]:
            if key in obj and isinstance(obj[key], list):
                return tuple(int(x) for x in obj[key])

    return None


def find_shape(
    feature_dir: Path,
    subdir_name: str,
    shard_id: int,
    explicit_shape: Optional[str] = None,
) -> Tuple[int, ...]:
    if explicit_shape:
        return tuple(int(x) for x in explicit_shape.split(","))

    candidates = [
        feature_dir / subdir_name / f"shard_{shard_id:03d}.meta.json",
        feature_dir / subdir_name / f"shard_{shard_id:03d}_fp16.meta.json",
        feature_dir / f"{subdir_name}_shape.json",
    ]

    for path in candidates:
        if path.exists():
            obj = load_json(path)
            shape = read_shape_from_json(obj)
            if shape is not None:
                return shape

    raise FileNotFoundError(
        f"Cannot infer shape for {subdir_name}. Tried: "
        + ", ".join(str(p) for p in candidates)
        + ". You can pass explicit shape with --patch_shape or --head_shape."
    )


def find_mmap_path(
    feature_dir: Path,
    subdir_name: str,
    shard_id: int,
    explicit_path: Optional[str] = None,
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    meta_candidates = [
        feature_dir / subdir_name / f"shard_{shard_id:03d}.meta.json",
        feature_dir / subdir_name / f"shard_{shard_id:03d}_fp16.meta.json",
    ]

    for meta_path in meta_candidates:
        if not meta_path.exists():
            continue
        meta = load_json(meta_path)
        for key in ["mmap_path", "output_path", "path"]:
            if key in meta:
                p = Path(meta[key])
                if not p.is_absolute():
                    candidates = [
                        ROOT / p,
                        feature_dir / p,
                        meta_path.parent / p,
                    ]
                else:
                    candidates = [p]
                for c in candidates:
                    if c.exists():
                        return c

    candidates = [
        feature_dir / subdir_name / f"shard_{shard_id:03d}_fp16.mmap",
        feature_dir / subdir_name / f"shard_{shard_id:03d}_float16.mmap",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Cannot find mmap for {subdir_name}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def find_images_jsonl(feature_dir: Path, shard_id: int, explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = [
        feature_dir / "patch_tokens" / f"shard_{shard_id:03d}.images.jsonl",
        feature_dir / "disentangled_self_attn" / f"shard_{shard_id:03d}.images.jsonl",
        feature_dir / "images.jsonl",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot find images jsonl. Tried: " + ", ".join(str(p) for p in candidates)
    )


def get_image_relpath(row: Dict[str, Any]) -> str:
    for key in [
        "path",
        "image_path",
        "file_path",
        "filepath",
        "file_name",
        "filename",
        "name",
    ]:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value

    # COCO-style fallback
    image_id = row.get("id", row.get("image_id", None))
    if image_id is not None:
        try:
            return f"COCO_train2014_{int(image_id):012d}.jpg"
        except Exception:
            pass

    raise KeyError(f"Cannot infer image path from row keys: {list(row.keys())}")


def resolve_image_path(row: Dict[str, Any], image_root: Path, feature_dir: Path) -> Path:
    rel = get_image_relpath(row)
    p = Path(rel)

    if p.is_absolute() and p.exists():
        return p

    candidates = []

    if not p.is_absolute():
        candidates.append(image_root / p)
        candidates.append(image_root / p.name)
        candidates.append(feature_dir / p)
        candidates.append(ROOT / p)

        # If rel contains train2014/xxx and image_root is already train2014,
        # basename fallback usually works.
        candidates.append(image_root / p.name)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Cannot resolve image path for {rel}. Tried: "
        + ", ".join(str(c) for c in candidates[:6])
    )


def normalize_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def scores_to_maps(scores: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    # scores: [H, P]
    if mode == "minmax":
        lo = scores.min(axis=1, keepdims=True)
        hi = scores.max(axis=1, keepdims=True)
        return (scores - lo) / np.maximum(hi - lo, 1e-6)

    if mode == "softmax":
        z = scores / temperature
        z = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z)
        prob = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)

        # For visualization, softmax values are usually too small.
        # Re-scale each head to [0, 1] for visible heatmaps.
        lo = prob.min(axis=1, keepdims=True)
        hi = prob.max(axis=1, keepdims=True)
        return (prob - lo) / np.maximum(hi - lo, 1e-12)

    if mode == "cosine":
        return np.clip((scores + 1.0) / 2.0, 0.0, 1.0)

    raise ValueError(f"Unknown normalize mode: {mode}")


def colorize(arr: np.ndarray, colormap: str) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)

    if mpl_cm is not None:
        cmap = mpl_cm.get_cmap(colormap)
        rgba = cmap(arr)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    gray = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    return ImageOps.colorize(gray, black="blue", white="red").convert("RGB")


def resize_keep_long_side(img: Image.Image, long_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) == long_side:
        return img
    scale = long_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BICUBIC)


def draw_label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    pad = 4
    draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return img


def make_grid(tiles: Sequence[Image.Image], ncols: Optional[int] = None, pad: int = 6) -> Image.Image:
    if not tiles:
        raise ValueError("No tiles to make grid.")

    h_count = len(tiles)
    if ncols is None or ncols <= 0:
        ncols = int(math.ceil(math.sqrt(h_count)))
    nrows = int(math.ceil(h_count / ncols))

    tile_w = max(t.size[0] for t in tiles)
    tile_h = max(t.size[1] for t in tiles)

    canvas_w = ncols * tile_w + (ncols + 1) * pad
    canvas_h = nrows * tile_h + (nrows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    for idx, tile in enumerate(tiles):
        r = idx // ncols
        c = idx % ncols
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + pad)

        if tile.size != (tile_w, tile_h):
            bg = Image.new("RGB", (tile_w, tile_h), color=(255, 255, 255))
            ox = (tile_w - tile.size[0]) // 2
            oy = (tile_h - tile.size[1]) // 2
            bg.paste(tile, (ox, oy))
            tile = bg

        canvas.paste(tile, (x, y))

    return canvas


def safe_stem_from_image_path(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("/", "_").replace("\\", "_")
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Feature directory, e.g. feature/vision/coco2014_dinov2_vitb14_reg_448",
    )
    parser.add_argument("--shard_id", type=int, default=0)

    parser.add_argument(
        "--disentangled_path",
        type=str,
        default=None,
        help="Optional explicit path to disentangled_self_attn shard mmap.",
    )
    parser.add_argument(
        "--patch_path",
        type=str,
        default=None,
        help="Optional explicit path to patch_tokens shard mmap.",
    )
    parser.add_argument(
        "--head_shape",
        type=str,
        default=None,
        help="Optional explicit disentangled shape, e.g. 82563,12,768",
    )
    parser.add_argument(
        "--patch_shape",
        type=str,
        default=None,
        help="Optional explicit patch shape, e.g. 82563,1024,768",
    )

    parser.add_argument(
        "--images_jsonl",
        type=str,
        default=None,
        help="Optional explicit images jsonl path.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Image root, e.g. data/mscoco14/train2014",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output visualization directory.",
    )

    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma/range indices, e.g. 0,5,10 or 0-99. If omitted, use first max_images or all.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Max number of images to visualize. If omitted and --indices omitted, visualize all.",
    )

    parser.add_argument(
        "--normalize",
        type=str,
        default="minmax",
        choices=["minmax", "softmax", "cosine"],
    )
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument(
        "--patch_grid",
        type=str,
        default=None,
        help="Patch grid H,W. If omitted, infer sqrt(P), e.g. 32,32.",
    )
    parser.add_argument(
        "--tile_long_side",
        type=int,
        default=224,
        help="Long side of each head tile in the grid.",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=0,
        help="Number of columns in head grid. 0 means auto.",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--colormap", type=str, default="turbo")

    parser.add_argument("--save_heatmap_grid", action="store_true")
    parser.add_argument("--save_overlay_grid", action="store_true")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)

    if not feature_dir.exists():
        raise FileNotFoundError(feature_dir)
    if not image_root.exists():
        raise FileNotFoundError(image_root)

    if not args.save_heatmap_grid and not args.save_overlay_grid:
        args.save_overlay_grid = True

    head_shape = find_shape(
        feature_dir=feature_dir,
        subdir_name="disentangled_self_attn",
        shard_id=args.shard_id,
        explicit_shape=args.head_shape,
    )
    patch_shape = find_shape(
        feature_dir=feature_dir,
        subdir_name="patch_tokens",
        shard_id=args.shard_id,
        explicit_shape=args.patch_shape,
    )

    if len(head_shape) != 3:
        raise ValueError(f"Expected disentangled shape [N,H,D], got {head_shape}")
    if len(patch_shape) != 3:
        raise ValueError(f"Expected patch shape [N,P,D], got {patch_shape}")

    num_images, num_heads, head_dim = head_shape
    patch_num_images, num_patches, patch_dim = patch_shape

    if num_images != patch_num_images:
        raise ValueError(
            f"N mismatch: disentangled N={num_images}, patch N={patch_num_images}"
        )
    if head_dim != patch_dim:
        raise ValueError(
            f"D mismatch: disentangled D={head_dim}, patch D={patch_dim}"
        )

    if args.patch_grid:
        gh, gw = [int(x) for x in args.patch_grid.split(",")]
    else:
        side = int(round(math.sqrt(num_patches)))
        if side * side != num_patches:
            raise ValueError(
                f"Cannot infer square patch grid from P={num_patches}. "
                f"Please pass --patch_grid H,W."
            )
        gh, gw = side, side

    if gh * gw != num_patches:
        raise ValueError(f"patch_grid {gh}x{gw} != num_patches {num_patches}")

    head_path = find_mmap_path(
        feature_dir=feature_dir,
        subdir_name="disentangled_self_attn",
        shard_id=args.shard_id,
        explicit_path=args.disentangled_path,
    )
    patch_path = find_mmap_path(
        feature_dir=feature_dir,
        subdir_name="patch_tokens",
        shard_id=args.shard_id,
        explicit_path=args.patch_path,
    )
    images_jsonl = find_images_jsonl(
        feature_dir=feature_dir,
        shard_id=args.shard_id,
        explicit_path=args.images_jsonl,
    )

    image_rows = load_jsonl(images_jsonl)
    if len(image_rows) < num_images:
        raise ValueError(
            f"images_jsonl has {len(image_rows)} rows, but features have {num_images} images."
        )

    indices = parse_indices(args.indices, num_images, args.max_images)

    heatmap_dir = output_dir / "heatmap_grid"
    overlay_dir = output_dir / "overlay_grid"
    if args.save_heatmap_grid:
        ensure_dir(heatmap_dir)
    if args.save_overlay_grid:
        ensure_dir(overlay_dir)

    print(f"[feature_dir] {feature_dir}")
    print(f"[head_path]   {head_path}")
    print(f"[patch_path]  {patch_path}")
    print(f"[images]      {images_jsonl}")
    print(f"[head_shape]  {head_shape}")
    print(f"[patch_shape] {patch_shape}")
    print(f"[patch_grid]  {gh}x{gw}")
    print(f"[num_images]  {len(indices)} selected / {num_images} total")
    print(f"[output_dir]  {output_dir}")

    head_mmap = np.memmap(
        head_path,
        dtype=np.float16,
        mode="r",
        shape=(num_images, num_heads, head_dim),
    )
    patch_mmap = np.memmap(
        patch_path,
        dtype=np.float16,
        mode="r",
        shape=(num_images, num_patches, patch_dim),
    )

    ncols = args.grid_cols if args.grid_cols and args.grid_cols > 0 else None

    for idx in tqdm(indices, desc="visualize disentangled heads"):
        row = image_rows[idx]
        image_path = resolve_image_path(row, image_root, feature_dir)

        img = Image.open(image_path).convert("RGB")
        img_tile = resize_keep_long_side(img, args.tile_long_side)
        tile_w, tile_h = img_tile.size

        heads = np.asarray(head_mmap[idx], dtype=np.float32)   # [H, D]
        patches = np.asarray(patch_mmap[idx], dtype=np.float32) # [P, D]

        heads = normalize_rows(heads)
        patches = normalize_rows(patches)

        # [H, D] @ [D, P] -> [H, P]
        scores = heads @ patches.T
        maps = scores_to_maps(scores, args.normalize, args.temperature) # [H, P]

        heatmap_tiles: List[Image.Image] = []
        overlay_tiles: List[Image.Image] = []

        for h in range(num_heads):
            attn = maps[h].reshape(gh, gw)
            attn_img = Image.fromarray((attn * 255).astype(np.uint8), mode="L")
            attn_img = attn_img.resize((tile_w, tile_h), Image.BICUBIC)
            attn_np = np.asarray(attn_img, dtype=np.float32) / 255.0

            heat = colorize(attn_np, args.colormap).resize((tile_w, tile_h), Image.BICUBIC)
            heat = draw_label(heat, f"head {h}")

            if args.save_heatmap_grid:
                heatmap_tiles.append(heat)

            if args.save_overlay_grid:
                overlay = Image.blend(img_tile, heat, alpha=args.alpha)
                overlay = draw_label(overlay, f"head {h}")
                overlay_tiles.append(overlay)

        stem = safe_stem_from_image_path(image_path)
        prefix = f"{idx:06d}_{stem}"

        if args.save_heatmap_grid:
            out_path = heatmap_dir / f"{prefix}_heads_heatmap.png"
            if args.force or not out_path.exists():
                grid = make_grid(heatmap_tiles, ncols=ncols)
                grid.save(out_path)

        if args.save_overlay_grid:
            out_path = overlay_dir / f"{prefix}_heads_overlay.png"
            if args.force or not out_path.exists():
                grid = make_grid(overlay_tiles, ncols=ncols)
                grid.save(out_path)

    print("[done]")


if __name__ == "__main__":
    main()
