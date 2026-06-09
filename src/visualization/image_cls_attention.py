#!/usr/bin/env python3
"""Visualize one image's DINOv2 CLS-token attention over patch tokens."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile

from src.features.dinov2 import CocoImageDataset, load_dinov2_model
from src.features.patch_store import find_image_row, load_json, load_jsonl, write_json

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_images(feature_dir: Path, image_root: Optional[Path]) -> List[Dict]:
    rows = sorted(load_jsonl(feature_dir / "images.jsonl"), key=lambda r: int(r.get("index", 0)))
    for row in rows:
        path = Path(str(row.get("path", "")))
        if not path.exists() and image_root is not None:
            candidate = image_root / str(row.get("file_name", ""))
            if candidate.exists():
                path = candidate
        row["path"] = str(path)
    return rows


def collect_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    raw_blocks = getattr(model, "blocks", None)
    if raw_blocks is None:
        raise AttributeError("DINOv2 model has no `blocks` attribute; cannot compute self-attention.")
    blocks: List[torch.nn.Module] = []

    def walk(module: torch.nn.Module) -> None:
        if hasattr(module, "attn") and hasattr(module, "norm1"):
            blocks.append(module)
            return
        for child in module.children():
            walk(child)

    for block in raw_blocks:
        walk(block)
    if not blocks:
        raise RuntimeError("Cannot find transformer blocks with `attn` and `norm1`.")
    return blocks


def prepare_tokens(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "prepare_tokens_with_masks"):
        return model.prepare_tokens_with_masks(images, masks=None)
    if hasattr(model, "prepare_tokens"):
        return model.prepare_tokens(images)
    raise AttributeError("DINOv2 model has no token preparation method.")


@torch.no_grad()
def compute_cls_patch_attention(model: torch.nn.Module, images: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    blocks = collect_transformer_blocks(model)
    x = prepare_tokens(model, images)
    for block in blocks[:-1]:
        x = block(x)

    last_block = blocks[-1]
    x = last_block.norm1(x)
    attn_module = last_block.attn
    if not hasattr(attn_module, "qkv") or not hasattr(attn_module, "num_heads"):
        raise AttributeError("Last attention block does not expose `qkv` and `num_heads`.")

    qkv = attn_module.qkv(x)
    batch, num_tokens, three_dim = qkv.shape
    if batch != 1:
        raise ValueError(f"Expected one image, got batch={batch}")

    num_heads = int(attn_module.num_heads)
    head_dim = three_dim // (3 * num_heads)
    qkv = qkv.reshape(batch, num_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    scale = float(getattr(attn_module, "scale", head_dim ** -0.5))
    attn = (q * scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)

    num_register_tokens = int(getattr(model, "num_register_tokens", 0))
    patch_start = 1 + num_register_tokens
    num_patches = grid_h * grid_w
    if num_tokens < patch_start + num_patches:
        raise ValueError(
            f"Token count {num_tokens} is too small for cls + {num_register_tokens} register + {num_patches} patch tokens."
        )

    cls_to_patch = attn[0, :, 0, patch_start: patch_start + num_patches].mean(dim=0)
    return cls_to_patch.float().cpu().numpy().reshape(grid_h, grid_w)


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    grid = grid.astype("float32", copy=False)
    lo, hi = float(grid.min()), float(grid.max())
    if hi <= lo:
        return np.zeros_like(grid, dtype="float32")
    return (grid - lo) / (hi - lo)


def simple_colormap(values: np.ndarray) -> np.ndarray:
    v = normalize_grid(values)
    r = (255.0 * v).astype(np.uint8)
    g = (255.0 * (1.0 - np.abs(v - 0.5) * 2.0)).astype(np.uint8)
    b = (255.0 * (1.0 - v)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def save_attention_images(image_path: Path, attention: np.ndarray, out_dir: Path, alpha: float) -> Tuple[Path, Path]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        heat = Image.fromarray(simple_colormap(attention), mode="RGB").resize(im.size, resample=resample)
        overlay = Image.blend(im, heat, alpha=alpha)

    heat_path = out_dir / "cls_attention_heatmap.png"
    overlay_path = out_dir / "cls_attention_overlay.png"
    heat.save(heat_path)
    overlay.save(overlay_path)
    return heat_path, overlay_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one image's DINOv2 CLS-to-patch self-attention map.")
    parser.add_argument("--feature_dir", type=str, default="feature/coco2014_dinov2_vitb14_reg_448")
    parser.add_argument("--image_name", type=str, default=None, help="Example: COCO_train2014_000000000009.jpg")
    parser.add_argument("--image_index", type=int, default=None, help="Optional exact image index from images.jsonl.")
    parser.add_argument("--image_root", type=str, default=None, help="Optional fallback root for original COCO train2014 images.")
    parser.add_argument("--model", type=str, default=None, help="Default: read from feature_dir/meta.json, fallback dinov2_vitb14.")
    parser.add_argument("--image_size", type=int, default=None, help="Default: read from feature_dir/meta.json, fallback 448.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    meta = load_json(feature_dir / "meta.json") if (feature_dir / "meta.json").exists() else {}
    grid = meta.get("patch_grid", [32, 32])
    grid_h, grid_w = int(grid[0]), int(grid[1])
    model_name = args.model or str(meta.get("model", "dinov2_vitb14"))
    image_size = int(args.image_size or meta.get("input_size", 448))

    image_root = Path(args.image_root) if args.image_root else None
    images = load_images(feature_dir, image_root)
    image_row = find_image_row(images, args.image_name, args.image_index)
    image_path = Path(str(image_row["path"]))
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot find image file: {image_path}. Use --image_root if paths moved.")

    stem = image_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else feature_dir / "vis" / "cls_attention" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = load_dinov2_model(model_name, device=device)
    image = CocoImageDataset([dict(image_row)], image_size=image_size)[0]["image"].unsqueeze(0).to(device)

    attention = compute_cls_patch_attention(model, image, grid_h=grid_h, grid_w=grid_w)
    npy_path = out_dir / "cls_attention.npy"
    np.save(npy_path, attention)
    heat_path, overlay_path = save_attention_images(image_path, attention, out_dir, alpha=args.overlay_alpha)

    config = {
        "feature_dir": str(feature_dir),
        "image": {
            "index": int(image_row["index"]),
            "image_id": image_row.get("image_id"),
            "file_name": image_row.get("file_name"),
            "path": str(image_path),
        },
        "model": model_name,
        "image_size": image_size,
        "grid": [grid_h, grid_w],
        "attention_source": "last_block_cls_query_to_patch_keys_mean_over_heads",
        "outputs": {
            "attention_npy": str(npy_path),
            "heatmap_png": str(heat_path),
            "overlay_png": str(overlay_path),
        },
    }
    write_json(out_dir / "config.json", config)
    print(f"[image] index={image_row['index']}, file_name={image_row.get('file_name')}")
    print(f"[save] {overlay_path}")
    print(f"[save] {heat_path}")
    print(f"[save] {npy_path}")


if __name__ == "__main__":
    main()
