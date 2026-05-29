#!/usr/bin/env python3
"""
Extract DINOv2 patch tokens for MSCOCO2014 train2014.

Default target used by this script:
  - dataset: ./coco/train2014 + ./coco/annotations/captions_train2014.json
  - model: dinov2_vitb14
  - resize: 448 x 448
  - patch tokens: [N, 1024, 768]
  - storage: fp16 sharded numpy memmap + json/jsonl metadata

This script is designed to be placed at the UNITRAN repository root.
It does not depend on pycocotools.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_coco_image_id(file_name: str) -> Optional[int]:
    match = re.search(r"_(\d{12})\.", file_name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", Path(file_name).stem)
    return int(match.group(1)) if match else None


def jsonl_dump(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def json_dump(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_default_ann(coco_root: Path, split: str) -> Optional[Path]:
    candidates = [
        coco_root / "annotations" / f"captions_{split}.json",
        coco_root / f"captions_{split}.json",
        coco_root / "annotations" / f"instances_{split}.json",
        coco_root / f"instances_{split}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_default_image_dir(coco_root: Path, split: str) -> Path:
    candidates = [coco_root / split, coco_root / "images" / split]
    for p in candidates:
        if p.exists():
            return p
    return coco_root / split


def build_image_rows(
    coco_root: Path,
    split: str,
    ann_path: Optional[Path],
    image_dir: Optional[Path],
    max_images: Optional[int] = None,
) -> List[Dict[str, Any]]:
    image_dir = image_dir or find_default_image_dir(coco_root, split)
    ann_path = ann_path or find_default_ann(coco_root, split)

    rows: List[Dict[str, Any]] = []

    if ann_path is not None and ann_path.exists():
        with ann_path.open("r", encoding="utf-8") as f:
            ann = json.load(f)
        images = sorted(ann.get("images", []), key=lambda x: x.get("file_name", ""))
        for idx, img in enumerate(images):
            file_name = img["file_name"]
            path = image_dir / file_name
            rows.append(
                {
                    "index": idx,
                    "image_id": int(img.get("id", parse_coco_image_id(file_name) or idx)),
                    "file_name": file_name,
                    "path": str(path),
                    "split": split,
                    "orig_height": int(img.get("height", -1)),
                    "orig_width": int(img.get("width", -1)),
                }
            )
    else:
        paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
        for idx, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    width, height = im.size
            except Exception:
                width, height = -1, -1
            rows.append(
                {
                    "index": idx,
                    "image_id": parse_coco_image_id(p.name) or idx,
                    "file_name": p.name,
                    "path": str(p),
                    "split": split,
                    "orig_height": int(height),
                    "orig_width": int(width),
                }
            )

    if not rows:
        raise FileNotFoundError(
            f"No images found. Checked coco_root={coco_root}, split={split}, "
            f"ann_path={ann_path}, image_dir={image_dir}"
        )

    if max_images is not None and max_images > 0:
        rows = rows[:max_images]
        for i, row in enumerate(rows):
            row["index"] = i

    return rows


class CocoImageDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], image_size: int):
        self.rows = list(rows)
        self.image_size = image_size
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def transform_image(self, im: Image.Image) -> torch.Tensor:
        # torchvision-free implementation to avoid version-specific torchvision binary issues.
        resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        im = im.resize((self.image_size, self.image_size), resample=resample)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image array, got shape={arr.shape}")
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, local_idx: int) -> Dict[str, Any]:
        row = self.rows[local_idx]
        path = row["path"]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                image = self.transform_image(im)
            ok = True
            err = ""
        except Exception as e:
            # Keep global tensor shape fixed. The failed image will be all-zero and marked in metadata.
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            ok = False
            err = repr(e)
        return {
            "image": image,
            "local_index": local_idx,
            "global_index": int(row["index"]),
            "ok": ok,
            "error": err,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "local_indices": [int(b["local_index"]) for b in batch],
        "global_indices": [int(b["global_index"]) for b in batch],
        "ok": [bool(b["ok"]) for b in batch],
        "errors": [str(b["error"]) for b in batch],
    }


def load_dinov2_model(model_name: str, device: torch.device) -> torch.nn.Module:
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    except Exception as e:
        raise RuntimeError(
            "Failed to load DINOv2 via torch.hub. If the server has no internet, "
            "pre-download facebookresearch/dinov2 or set TORCH_HOME to a cache containing the model. "
            f"Original error: {repr(e)}"
        ) from e
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_patch_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model.forward_features(images)
    else:
        out = model.forward_features(images)

    if isinstance(out, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "patchtokens"):
            if key in out:
                return out[key]
        raise KeyError(f"Cannot find patch tokens in DINOv2 output keys: {list(out.keys())}")

    if isinstance(out, (tuple, list)):
        # Defensive fallback; DINOv2 normally returns dict for forward_features.
        for item in out:
            if torch.is_tensor(item) and item.ndim == 3:
                return item

    raise TypeError(f"Unsupported DINOv2 forward_features output type: {type(out)}")


def shard_rows(rows: Sequence[Dict[str, Any]], shard_id: int, num_shards: int, strategy: str) -> List[Dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must satisfy 0 <= shard_id < num_shards, got {shard_id}/{num_shards}")

    if strategy == "mod":
        return [r for i, r in enumerate(rows) if i % num_shards == shard_id]
    if strategy == "contiguous":
        n = len(rows)
        start = (n * shard_id) // num_shards
        end = (n * (shard_id + 1)) // num_shards
        return list(rows[start:end])
    raise ValueError(f"Unknown shard strategy: {strategy}")


def make_enriched_row(row: Dict[str, Any], args: argparse.Namespace, local_i: int, ok: bool, error: str) -> Dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "local_shard_index": int(local_i),
            "resize_size": [int(args.image_size), int(args.image_size)],
            "patch_size": int(args.patch_size),
            "patch_grid": [int(args.grid_size), int(args.grid_size)],
            "num_patches": int(args.num_patches),
            "feature_dim": int(args.feature_dim),
            "feature_dtype": args.save_dtype,
            "status": "ok" if ok else "failed_image_load_zero_filled",
            "error": error,
        }
    )
    return out


def write_global_files(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    out_dir = Path(args.out_dir)
    meta = {
        "dataset": "mscoco2014",
        "split": args.split,
        "num_images": len(rows),
        "coco_root": str(Path(args.coco_root).resolve()),
        "image_dir": str(Path(args.image_dir).resolve()) if args.image_dir else str(find_default_image_dir(Path(args.coco_root), args.split).resolve()),
        "ann_path": str(Path(args.ann_path).resolve()) if args.ann_path else str(find_default_ann(Path(args.coco_root), args.split) or ""),
        "model": args.model,
        "input_size": args.image_size,
        "patch_size": args.patch_size,
        "patch_grid": [args.grid_size, args.grid_size],
        "num_patches_per_image": args.num_patches,
        "feature_dim": args.feature_dim,
        "feature_dtype": args.save_dtype,
        "feature_storage": "sharded_numpy_memmap",
        "keep_full_patch_tokens": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_dump(out_dir / "meta.json", meta)

    shape = {
        "storage": "sharded_numpy_memmap",
        "full_shape": [len(rows), args.num_patches, args.feature_dim],
        "dtype": args.save_dtype,
        "note": "Read individual patch_tokens/shard_*.meta.json files for per-shard shapes and global indices.",
    }
    json_dump(out_dir / "patch_tokens_shape.json", shape)

    enriched = []
    for r in rows:
        rr = dict(r)
        rr.update(
            {
                "resize_size": [args.image_size, args.image_size],
                "patch_size": args.patch_size,
                "patch_grid": [args.grid_size, args.grid_size],
                "num_patches": args.num_patches,
                "feature_dim": args.feature_dim,
                "feature_dtype": args.save_dtype,
            }
        )
        enriched.append(rr)
    jsonl_dump(out_dir / "images.jsonl", enriched)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DINOv2 patch tokens for COCO2014 train2014.")
    parser.add_argument("--coco_root", type=str, default="./coco", help="COCO root, default: ./coco")
    parser.add_argument("--split", type=str, default="train2014", help="COCO split, default: train2014")
    parser.add_argument("--ann_path", type=str, default=None, help="Optional captions_train2014.json / instances_train2014.json")
    parser.add_argument("--image_dir", type=str, default=None, help="Optional image dir, e.g. ./coco/train2014")
    parser.add_argument("--out_dir", type=str, default="feature/coco2014_dinov2_vitb14_448")
    parser.add_argument("--model", type=str, default="dinov2_vitb14", choices=sorted(MODEL_DIMS.keys()))
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_strategy", type=str, default="contiguous", choices=["contiguous", "mod"])
    parser.add_argument("--max_images", type=int, default=None, help="Debug only. Use first N images before sharding.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16", "none"])
    parser.add_argument("--save_image_mean_pt", action="store_true", help="Also save image-level mean patch features as a UNITRAN-friendly .pt file.")
    parser.add_argument("--normalize_image_mean", action="store_true", help="L2 normalize image mean features before saving optional .pt.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing shard files.")
    parser.add_argument("--write_global_meta", action="store_true", help="Write meta.json/images.jsonl/patch_tokens_shape.json. Safe to run on shard 0.")
    args = parser.parse_args()

    set_seed(args.seed)

    args.grid_size = args.image_size // args.patch_size
    if args.image_size % args.patch_size != 0:
        raise ValueError(f"image_size must be divisible by patch_size, got {args.image_size}/{args.patch_size}")
    args.num_patches = args.grid_size * args.grid_size
    args.feature_dim = MODEL_DIMS[args.model]

    out_dir = Path(args.out_dir)
    patch_dir = out_dir / "patch_tokens"
    patch_dir.mkdir(parents=True, exist_ok=True)

    rows = build_image_rows(
        coco_root=Path(args.coco_root),
        split=args.split,
        ann_path=Path(args.ann_path) if args.ann_path else None,
        image_dir=Path(args.image_dir) if args.image_dir else None,
        max_images=args.max_images,
    )

    if args.write_global_meta or args.shard_id == 0:
        # shard 0 writes global metadata by default. Content is deterministic and idempotent.
        write_global_files(args, rows)

    shard = shard_rows(rows, args.shard_id, args.num_shards, args.shard_strategy)
    if not shard:
        raise RuntimeError(f"Shard {args.shard_id}/{args.num_shards} is empty.")

    dtype = np.float16 if args.save_dtype == "float16" else np.float32
    shard_prefix = f"shard_{args.shard_id:03d}"
    mmap_path = patch_dir / f"{shard_prefix}_fp16.mmap" if args.save_dtype == "float16" else patch_dir / f"{shard_prefix}_fp32.mmap"
    meta_path = patch_dir / f"{shard_prefix}.meta.json"
    image_jsonl_path = patch_dir / f"{shard_prefix}.images.jsonl"
    done_path = patch_dir / f"{shard_prefix}.done"

    if done_path.exists() and not args.overwrite:
        print(f"[skip] {done_path} exists. Use --overwrite to rerun.")
        return
    if mmap_path.exists() and not args.overwrite:
        raise FileExistsError(f"{mmap_path} exists. Use --overwrite or remove it.")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[device] {device}")
    print(f"[data] total_images={len(rows)}, shard_images={len(shard)}, shard_id={args.shard_id}/{args.num_shards}")
    print(f"[shape] per_image=[{args.num_patches}, {args.feature_dim}], grid={args.grid_size}x{args.grid_size}")
    print(f"[save] {mmap_path}")

    model = load_dinov2_model(args.model, device=device)

    dataset = CocoImageDataset(shard, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )

    mmap = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(len(shard), args.num_patches, args.feature_dim))

    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16

    image_rows_out: List[Dict[str, Any]] = []
    mean_feats: List[torch.Tensor] = []
    num_failed = 0
    start_time = time.time()

    for batch in tqdm(loader, desc=f"extract shard {args.shard_id:03d}"):
        images = batch["images"].to(device, non_blocking=True)
        patch_tokens = extract_patch_tokens(model, images, use_amp=use_amp, amp_dtype=amp_dtype)

        if patch_tokens.ndim != 3:
            raise RuntimeError(f"Expected patch tokens [B,P,D], got {tuple(patch_tokens.shape)}")
        if patch_tokens.shape[1] != args.num_patches or patch_tokens.shape[2] != args.feature_dim:
            raise RuntimeError(
                f"Unexpected patch token shape {tuple(patch_tokens.shape)}; "
                f"expected [B,{args.num_patches},{args.feature_dim}]."
            )

        patch_cpu = patch_tokens.detach().cpu().to(torch.float16 if args.save_dtype == "float16" else torch.float32).numpy()

        failed_positions = [j for j, ok in enumerate(batch["ok"]) if not ok]
        if failed_positions:
            # Do not keep features produced from zero placeholder images.
            # Mark them in metadata and store zero patch tokens.
            patch_cpu[failed_positions] = 0

        for j, local_i in enumerate(batch["local_indices"]):
            mmap[local_i] = patch_cpu[j]
            ok = bool(batch["ok"][j])
            err = str(batch["errors"][j])
            if not ok:
                num_failed += 1
            image_rows_out.append(make_enriched_row(shard[local_i], args, local_i, ok, err))

        if args.save_image_mean_pt:
            means_np = patch_cpu.astype(np.float32).mean(axis=1)
            means = torch.from_numpy(means_np)
            if args.normalize_image_mean:
                means = torch.nn.functional.normalize(means, dim=-1)
            mean_feats.append(means.to(torch.float16 if args.save_dtype == "float16" else torch.float32))

    mmap.flush()

    jsonl_dump(image_jsonl_path, sorted(image_rows_out, key=lambda r: r["local_shard_index"]))

    shard_meta = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "shard_strategy": args.shard_strategy,
        "num_images": len(shard),
        "global_indices": [int(r["index"]) for r in shard],
        "mmap_path": str(mmap_path),
        "shape": [len(shard), args.num_patches, args.feature_dim],
        "dtype": args.save_dtype,
        "model": args.model,
        "input_size": args.image_size,
        "patch_size": args.patch_size,
        "patch_grid": [args.grid_size, args.grid_size],
        "num_failed_images": num_failed,
        "elapsed_sec": round(time.time() - start_time, 3),
    }
    json_dump(meta_path, shard_meta)

    if args.save_image_mean_pt:
        image_mean_dir = out_dir / "image_mean_feats"
        image_mean_dir.mkdir(parents=True, exist_ok=True)
        mean_path = image_mean_dir / f"{shard_prefix}_image_mean_{args.save_dtype}.pt"
        vision_feats = torch.cat(mean_feats, dim=0) if mean_feats else torch.empty(0, args.feature_dim)
        payload = {
            "vision_feats": vision_feats,
            "global_indices": torch.tensor([int(r["index"]) for r in shard], dtype=torch.long),
            "file_names": [r["file_name"] for r in shard],
            "model": args.model,
            "input_size": args.image_size,
            "feature_source": "mean_of_dinov2_patch_tokens",
            "normalized": bool(args.normalize_image_mean),
        }
        torch.save(payload, mean_path)
        print(f"[save] {mean_path}")

    done_path.write_text("done\n", encoding="utf-8")
    print(f"[done] shard={args.shard_id}, images={len(shard)}, failed={num_failed}, elapsed={time.time() - start_time:.1f}s")
    print(f"[save] {meta_path}")
    print(f"[save] {image_jsonl_path}")
    print(f"[save] {done_path}")


if __name__ == "__main__":
    main()
