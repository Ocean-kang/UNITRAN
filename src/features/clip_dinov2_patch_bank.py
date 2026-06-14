#!/usr/bin/env python3
"""Extract a paired CLIP-V -> DINOv2 patch-token bank for mapping training.

Output .pt keys:
  clip_feats   [num_images * num_patches, clip_dim]
  dinov2_feats [num_images * num_patches, dinov2_dim]

Pairing rule: same image, same resized square input, same patch index.
For a one-to-one patch grid, use a CLIP ViT with the same patch size as DINOv2,
for example openai/clip-vit-large-patch14 with image_size=224.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.features.dinov2 import (
    MODEL_DIMS,
    build_image_rows,
    extract_patch_tokens as extract_dinov2_patch_tokens,
    infer_coco_dataset_name,
    load_dinov2_model,
    shard_rows,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

DINO_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
DINO_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32).view(3, 1, 1)
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32).view(3, 1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


class RawSquareImageDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], image_size: int):
        self.rows = list(rows)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        try:
            with Image.open(row["path"]) as im:
                im = im.convert("RGB")
                resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
                im = im.resize((self.image_size, self.image_size), resample=resample)
                arr = np.asarray(im, dtype=np.float32) / 255.0
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected RGB image array, got shape={arr.shape}")
            image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            ok, err = True, ""
        except Exception as e:
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            ok, err = False, repr(e)
        return {"image": image, "row": row, "ok": ok, "error": err}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "rows": [b["row"] for b in batch],
        "ok": [bool(b["ok"]) for b in batch],
        "errors": [str(b["error"]) for b in batch],
    }


def normalize_images(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=images.device, dtype=images.dtype)
    std = std.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def load_clip_vision_model(model_name_or_path: str, device: torch.device) -> torch.nn.Module:
    try:
        from transformers import CLIPVisionModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for CLIP-V extraction. It is already listed in conda.yaml.") from e
    model = CLIPVisionModel.from_pretrained(model_name_or_path)
    return model.eval().to(device)


def clip_grid_patch_dim(model: torch.nn.Module, image_size: int) -> Tuple[int, int, int]:
    cfg = model.config
    patch_size = int(getattr(cfg, "patch_size"))
    cfg_image_size = int(getattr(cfg, "image_size"))
    hidden_size = int(getattr(cfg, "hidden_size"))
    if image_size != cfg_image_size:
        raise ValueError(
            f"CLIP config image_size={cfg_image_size}, but image_size={image_size}. "
            "Use openai/clip-vit-large-patch14 for 224 or openai/clip-vit-large-patch14-336 for 336."
        )
    return image_size // patch_size, patch_size, hidden_size


@torch.no_grad()
def extract_clip_patch_tokens(model: torch.nn.Module, pixel_values: torch.Tensor, use_amp: bool, amp_dtype: torch.dtype) -> torch.Tensor:
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(pixel_values=pixel_values, return_dict=True)
    else:
        out = model(pixel_values=pixel_values, return_dict=True)
    return out.last_hidden_state[:, 1:, :]


def l2_normalize(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1, eps=1e-12) if enabled else x


def flatten_and_sample(clip_tokens: torch.Tensor, dino_tokens: torch.Tensor, max_pairs: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    clip_flat = clip_tokens.reshape(-1, clip_tokens.shape[-1]).contiguous()
    dino_flat = dino_tokens.reshape(-1, dino_tokens.shape[-1]).contiguous()
    if max_pairs > 0 and clip_flat.shape[0] > max_pairs:
        gen = torch.Generator().manual_seed(seed)
        keep = torch.randperm(clip_flat.shape[0], generator=gen)[:max_pairs]
        clip_flat = clip_flat[keep]
        dino_flat = dino_flat[keep]
    return clip_flat, dino_flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paired CLIP-V and DINOv2 patch tokens.")
    parser.add_argument("--coco_root", type=str, default="data/mscoco14")
    parser.add_argument("--split", type=str, default="train2014")
    parser.add_argument("--ann_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="feature/clip_dinov2/coco2014_clipL14_dinov2_vitb14_reg_224")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitb14_reg", choices=sorted(MODEL_DIMS.keys()))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--dinov2_patch_size", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_strategy", type=str, default="contiguous", choices=["contiguous", "mod"])
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_pairs", type=int, default=0, help="0 keeps all patch pairs from this run.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16", "none"])
    parser.add_argument("--normalize_features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.image_size % args.dinov2_patch_size != 0:
        raise ValueError(f"image_size must be divisible by DINOv2 patch size: {args.image_size}/{args.dinov2_patch_size}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_name = "paired_patch_bank.pt" if args.num_shards == 1 else f"paired_patch_bank_shard_{args.shard_id:03d}.pt"
    pt_path = out_dir / pt_name
    if pt_path.exists() and not args.overwrite:
        raise FileExistsError(f"{pt_path} exists. Use --overwrite or remove it.")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16

    rows = build_image_rows(
        coco_root=Path(args.coco_root),
        split=args.split,
        ann_path=Path(args.ann_path) if args.ann_path else None,
        image_dir=Path(args.image_dir) if args.image_dir else None,
        max_images=args.max_images,
    )
    shard = shard_rows(rows, args.shard_id, args.num_shards, args.shard_strategy)
    if not shard:
        raise RuntimeError(f"Shard {args.shard_id}/{args.num_shards} is empty.")

    print(f"[device] {device}")
    print(f"[data] total_images={len(rows)}, shard_images={len(shard)}, shard_id={args.shard_id}/{args.num_shards}")
    print(f"[models] clip={args.clip_model}, dinov2={args.dinov2_model}")

    clip_model = load_clip_vision_model(args.clip_model, device)
    dinov2_model = load_dinov2_model(args.dinov2_model, device)
    clip_grid, clip_patch_size, clip_dim = clip_grid_patch_dim(clip_model, args.image_size)
    dino_grid = args.image_size // args.dinov2_patch_size
    dino_dim = MODEL_DIMS[args.dinov2_model]
    if clip_grid != dino_grid:
        raise ValueError(
            f"Patch grid mismatch: CLIP={clip_grid}x{clip_grid}, DINOv2={dino_grid}x{dino_grid}. "
            "For direct same-index pairing, choose matching patch grids."
        )
    num_patches = dino_grid * dino_grid
    print(f"[shape] grid={dino_grid}x{dino_grid}, patches/image={num_patches}, clip_dim={clip_dim}, dinov2_dim={dino_dim}")

    loader = DataLoader(
        RawSquareImageDataset(shard, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )

    clip_chunks: List[torch.Tensor] = []
    dino_chunks: List[torch.Tensor] = []
    out_rows: List[Dict[str, Any]] = []
    failed = 0
    start = time.time()

    for batch in tqdm(loader, desc="extract paired patches"):
        raw = batch["images"].to(device, non_blocking=True)
        clip_in = normalize_images(raw, CLIP_MEAN, CLIP_STD)
        dino_in = normalize_images(raw, DINO_MEAN, DINO_STD)

        clip_tokens = extract_clip_patch_tokens(clip_model, clip_in, use_amp, amp_dtype).float()
        dino_tokens = extract_dinov2_patch_tokens(dinov2_model, dino_in, use_amp, amp_dtype).float()
        if tuple(clip_tokens.shape[1:]) != (num_patches, clip_dim):
            raise RuntimeError(f"Unexpected CLIP patch shape: {tuple(clip_tokens.shape)}")
        if tuple(dino_tokens.shape[1:]) != (num_patches, dino_dim):
            raise RuntimeError(f"Unexpected DINOv2 patch shape: {tuple(dino_tokens.shape)}")

        clip_tokens = l2_normalize(clip_tokens, args.normalize_features)
        dino_tokens = l2_normalize(dino_tokens, args.normalize_features)
        bad = [i for i, ok in enumerate(batch["ok"]) if not ok]
        if bad:
            clip_tokens[bad] = 0
            dino_tokens[bad] = 0

        clip_chunks.append(clip_tokens.detach().cpu().to(save_dtype))
        dino_chunks.append(dino_tokens.detach().cpu().to(save_dtype))

        for row, ok, err in zip(batch["rows"], batch["ok"], batch["errors"]):
            r = dict(row)
            r.update({"status": "ok" if ok else "failed_image_load_zero_filled", "error": err})
            out_rows.append(r)
            failed += int(not ok)

    clip_all = torch.cat(clip_chunks, dim=0)
    dino_all = torch.cat(dino_chunks, dim=0)
    clip_flat, dino_flat = flatten_and_sample(clip_all, dino_all, args.max_pairs, args.seed)

    payload = {
        "clip_feats": clip_flat,
        "dinov2_feats": dino_flat,
        "global_indices": torch.tensor([int(r["index"]) for r in shard], dtype=torch.long),
        "file_names": [r["file_name"] for r in shard],
        "source_key": "clip_feats",
        "target_key": "dinov2_feats",
        "clip_model": args.clip_model,
        "dinov2_model": args.dinov2_model,
        "image_size": int(args.image_size),
        "clip_patch_size": int(clip_patch_size),
        "dinov2_patch_size": int(args.dinov2_patch_size),
        "patch_grid": [int(dino_grid), int(dino_grid)],
        "num_patches_per_image": int(num_patches),
        "pairing": "same_image_same_resized_square_same_patch_index",
        "normalized": bool(args.normalize_features),
        "max_pairs": int(args.max_pairs),
    }
    torch.save(payload, pt_path)

    write_jsonl(out_dir / "images.jsonl", out_rows)
    write_json(out_dir / "meta.json", {
        "dataset": infer_coco_dataset_name(Path(args.coco_root), args.split),
        "split": args.split,
        "num_images": len(shard),
        "num_failed_images": failed,
        "clip_model": args.clip_model,
        "dinov2_model": args.dinov2_model,
        "image_size": args.image_size,
        "patch_grid": [dino_grid, dino_grid],
        "num_patches_per_image": num_patches,
        "clip_dim": clip_dim,
        "dinov2_dim": dino_dim,
        "feature_dtype": args.save_dtype,
        "paired_pt_path": str(pt_path),
        "paired_pt_shape": {"clip_feats": list(clip_flat.shape), "dinov2_feats": list(dino_flat.shape)},
        "elapsed_sec": round(time.time() - start, 3),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"[save] {pt_path}")
    print(f"[done] images={len(shard)}, failed={failed}, pairs={len(clip_flat)}, elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
