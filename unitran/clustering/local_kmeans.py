#!/usr/bin/env python3
"""
Build per-image local KMeans centers from extracted DINOv2 COCO patch tokens.

Input expected from the DINOv2 extraction step:

feature/coco2014_dinov2_vitb14_448/
  images.jsonl
  meta.json
  patch_tokens_shape.json
  patch_tokens/
    shard_000_fp16.mmap
    shard_000.meta.json
    ...

Output is another feature directory with the same sharded memmap layout, but
patch_tokens now stores local centers per image:

feature/coco2014_dinov2_vitb14_448_local_k20/
  images.jsonl
  meta.json
  patch_tokens_shape.json
  patch_tokens/
    shard_000_fp16.mmap  # shape [num_images_in_shard, 20, dim]
    shard_000.meta.json
    ...

The output directory can be passed directly to tools/cluster.py patch-global
for the second-stage global KMeans.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


@dataclass
class ShardInfo:
    shard_id: int
    meta_path: Path
    mmap_path: Path
    shape: Tuple[int, int, int]
    dtype: str
    global_indices: np.ndarray

    @property
    def num_images(self) -> int:
        return self.shape[0]

    @property
    def num_patches(self) -> int:
        return self.shape[1]

    @property
    def dim(self) -> int:
        return self.shape[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def jsonl_dump(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
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


def load_shards(feature_dir: Path, max_images: Optional[int] = None) -> List[ShardInfo]:
    patch_dir = feature_dir / "patch_tokens"
    if not patch_dir.exists():
        raise FileNotFoundError(f"Cannot find patch token dir: {patch_dir}")

    meta_paths = sorted(patch_dir.glob("shard_*.meta.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No shard_*.meta.json found under {patch_dir}")

    shards: List[ShardInfo] = []
    remaining = max_images

    for meta_path in meta_paths:
        meta = load_json(meta_path)
        shape = tuple(int(x) for x in meta["shape"])
        dtype = str(meta.get("dtype", "float16"))
        global_indices = np.asarray(meta.get("global_indices", list(range(shape[0]))), dtype=np.int64)

        mmap_value = meta.get("mmap_path", "")
        mmap_path = Path(mmap_value)
        if not mmap_path.is_absolute():
            candidate = feature_dir / mmap_value
            mmap_path = candidate if candidate.exists() else patch_dir / Path(mmap_value).name
        if not mmap_path.exists():
            suffix = "fp16" if dtype == "float16" else "fp32"
            mmap_path = patch_dir / f"shard_{int(meta.get('shard_id', len(shards))):03d}_{suffix}.mmap"
        if not mmap_path.exists():
            raise FileNotFoundError(f"Cannot locate mmap for {meta_path}: {mmap_path}")

        if remaining is not None:
            if remaining <= 0:
                break
            keep = min(shape[0], remaining)
            shape = (keep, shape[1], shape[2])
            global_indices = global_indices[:keep]
            remaining -= keep

        shards.append(
            ShardInfo(
                shard_id=int(meta.get("shard_id", len(shards))),
                meta_path=meta_path,
                mmap_path=mmap_path,
                shape=shape,
                dtype=dtype,
                global_indices=global_indices,
            )
        )

    if not shards:
        raise RuntimeError("No usable shards loaded.")

    num_patches = shards[0].num_patches
    dim = shards[0].dim
    for shard in shards:
        if shard.num_patches != num_patches or shard.dim != dim:
            raise ValueError(
                f"Shard shape mismatch: {shard.meta_path} shape={shard.shape}, "
                f"expected P={num_patches}, D={dim}"
            )
    return shards


def open_mmap(shard: ShardInfo, mode: str = "r") -> np.memmap:
    np_dtype = np.float16 if shard.dtype == "float16" else np.float32
    return np.memmap(shard.mmap_path, dtype=np_dtype, mode=mode, shape=shard.shape)


def to_device_images(
    arr: np.ndarray,
    device: torch.device,
    metric: str,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    x = np.array(arr, copy=True, order="C")
    t = torch.from_numpy(x).to(device=device, non_blocking=True)
    if metric == "cosine":
        t = F.normalize(t.float(), dim=-1, eps=1e-12)
    else:
        t = t.float()
    if compute_dtype in (torch.float16, torch.bfloat16) and device.type == "cuda":
        t = t.to(compute_dtype)
    return t


@torch.no_grad()
def batched_local_kmeans(
    x: torch.Tensor,
    local_k: int,
    num_iters: int,
    metric: str,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Run independent KMeans for each image in a batch.

    x: [B, P, D]. Return centers: [B, local_k, D] in fp32.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [B,P,D], got {tuple(x.shape)}")

    b, p, d = x.shape
    if p < local_k:
        raise ValueError(f"local_k must be <= patches per image, got local_k={local_k}, patches={p}")

    init_idx = np.stack([rng.choice(p, size=local_k, replace=False) for _ in range(b)], axis=0)
    init_idx_t = torch.from_numpy(init_idx).to(device=x.device, dtype=torch.long)
    batch_idx = torch.arange(b, device=x.device).view(b, 1)
    centers = x[batch_idx, init_idx_t].float().contiguous()
    if metric == "cosine":
        centers = F.normalize(centers, dim=-1, eps=1e-12)

    for _ in range(num_iters):
        x_f = x.float()
        if metric == "cosine":
            sims = torch.bmm(x.to(centers.dtype), centers.transpose(1, 2))
            labels = sims.argmax(dim=2)
        elif metric == "l2":
            x2 = (x_f * x_f).sum(dim=2, keepdim=True)
            c2 = (centers * centers).sum(dim=2).unsqueeze(1)
            dist = x2 + c2 - 2.0 * torch.bmm(x_f, centers.transpose(1, 2))
            labels = dist.argmin(dim=2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        new_centers = torch.zeros((b, local_k, d), dtype=torch.float32, device=x.device)
        counts = torch.zeros((b, local_k), dtype=torch.float32, device=x.device)
        new_centers.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, d), x_f)
        counts.scatter_add_(1, labels, torch.ones((b, p), dtype=torch.float32, device=x.device))

        updated = new_centers / counts.clamp_min(1.0).unsqueeze(-1)
        updated = torch.where(counts.unsqueeze(-1) > 0, updated, centers)
        if metric == "cosine":
            updated = F.normalize(updated, dim=-1, eps=1e-12)
        centers = updated.contiguous()

    return centers.float()


def make_output_dir(feature_dir: Path, out_dir: Optional[str], local_k: int) -> Path:
    if out_dir:
        return Path(out_dir)
    return feature_dir.parent / f"{feature_dir.name}_local_k{local_k}"


def write_global_files(
    args: argparse.Namespace,
    feature_dir: Path,
    out_dir: Path,
    shards: Sequence[ShardInfo],
) -> None:
    source_meta = load_json(feature_dir / "meta.json") if (feature_dir / "meta.json").exists() else {}
    source_num_patches = int(shards[0].num_patches)
    total_images = int(sum(s.num_images for s in shards))

    meta = dict(source_meta)
    meta.update(
        {
            "num_images": total_images,
            "num_patches_per_image": int(args.local_k),
            "source_num_patches_per_image": source_num_patches,
            "feature_dim": int(shards[0].dim),
            "feature_dtype": args.save_dtype,
            "feature_storage": "sharded_numpy_memmap",
            "feature_source": "per_image_local_kmeans_centers_of_dinov2_patch_tokens",
            "source_feature_dir": str(feature_dir),
            "local_precluster_k": int(args.local_k),
            "local_precluster_num_iters": int(args.local_iters),
            "local_precluster_metric": args.metric,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    json_dump(out_dir / "meta.json", meta)

    shape = {
        "storage": "sharded_numpy_memmap",
        "full_shape": [total_images, int(args.local_k), int(shards[0].dim)],
        "dtype": args.save_dtype,
        "source_full_shape": [total_images, source_num_patches, int(shards[0].dim)],
        "note": "Each per-image row stores local KMeans centers, not original spatial patch tokens.",
    }
    json_dump(out_dir / "patch_tokens_shape.json", shape)

    images_path = feature_dir / "images.jsonl"
    if images_path.exists():
        rows_by_index = {int(r.get("index", i)): r for i, r in enumerate(load_jsonl(images_path))}
        rows: List[Dict[str, Any]] = []
        for shard in shards:
            for global_i in shard.global_indices:
                row = dict(rows_by_index.get(int(global_i), {"index": int(global_i)}))
                row["num_patches"] = int(args.local_k)
                row["source_num_patches"] = source_num_patches
                row["feature_source"] = "per_image_local_kmeans_centers_of_dinov2_patch_tokens"
                rows.append(row)
        jsonl_dump(out_dir / "images.jsonl", sorted(rows, key=lambda r: int(r.get("index", 0))))


def build_local_centers(args: argparse.Namespace) -> None:
    feature_dir = Path(args.feature_dir)
    out_dir = make_output_dir(feature_dir, args.out_dir, args.local_k)
    patch_dir = out_dir / "patch_tokens"
    patch_dir.mkdir(parents=True, exist_ok=True)

    shards = load_shards(feature_dir, max_images=args.max_images)
    if args.local_k > shards[0].num_patches:
        raise ValueError(f"--local_k={args.local_k} is larger than patches per image={shards[0].num_patches}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    compute_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.compute_dtype]
    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    suffix = "fp16" if args.save_dtype == "float16" else "fp32"
    rng = np.random.default_rng(args.seed)

    print(f"[local-centers] source={feature_dir}")
    print(f"[local-centers] output={out_dir}")
    print(f"[local-centers] device={device}, metric={args.metric}, local_k={args.local_k}, local_iters={args.local_iters}")
    print(f"[local-centers] images={sum(s.num_images for s in shards)}, source_patches={shards[0].num_patches}, dim={shards[0].dim}")

    write_global_files(args, feature_dir, out_dir, shards)

    start_time = time.time()
    for shard in shards:
        shard_prefix = f"shard_{shard.shard_id:03d}"
        mmap_path = patch_dir / f"{shard_prefix}_{suffix}.mmap"
        meta_path = patch_dir / f"{shard_prefix}.meta.json"
        done_path = patch_dir / f"{shard_prefix}.done"

        if done_path.exists() and not args.overwrite:
            print(f"[skip] {done_path} exists. Use --overwrite to rerun.")
            continue
        if mmap_path.exists() and not args.overwrite:
            raise FileExistsError(f"{mmap_path} exists. Use --overwrite or remove it.")

        out_mmap = np.memmap(mmap_path, dtype=save_dtype, mode="w+", shape=(shard.num_images, args.local_k, shard.dim))
        source_mmap = open_mmap(shard, mode="r")

        for start in tqdm(range(0, shard.num_images, args.chunk_images), desc=f"local kmeans shard {shard.shard_id:03d}"):
            end = min(start + args.chunk_images, shard.num_images)
            arr = np.asarray(source_mmap[start:end])
            x = to_device_images(arr, device=device, metric=args.metric, compute_dtype=compute_dtype)
            centers = batched_local_kmeans(
                x=x,
                local_k=args.local_k,
                num_iters=args.local_iters,
                metric=args.metric,
                rng=rng,
            )
            out_mmap[start:end] = centers.detach().cpu().numpy().astype(save_dtype, copy=False)
            del x, centers

        out_mmap.flush()
        del out_mmap, source_mmap

        shard_meta = {
            "shard_id": int(shard.shard_id),
            "num_images": int(shard.num_images),
            "global_indices": [int(x) for x in shard.global_indices],
            "mmap_path": str(mmap_path),
            "shape": [int(shard.num_images), int(args.local_k), int(shard.dim)],
            "dtype": args.save_dtype,
            "feature_source": "per_image_local_kmeans_centers_of_dinov2_patch_tokens",
            "source_mmap_path": str(shard.mmap_path),
            "source_shape": list(shard.shape),
            "local_precluster_k": int(args.local_k),
            "local_precluster_num_iters": int(args.local_iters),
            "local_precluster_metric": args.metric,
        }
        json_dump(meta_path, shard_meta)
        done_path.write_text("done\n", encoding="utf-8")
        print(f"[save] {mmap_path}")
        print(f"[save] {meta_path}")

    print(f"[done] elapsed={time.time() - start_time:.1f}s")
    print(f"[next] python tools/cluster.py patch-global --feature_dir {out_dir} --k <GLOBAL_K> --num_iters 20")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-image local KMeans centers from COCO DINOv2 patch tokens.")
    parser.add_argument("--feature_dir", type=str, default="feature/coco2014_dinov2_vitb14_448")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--local_k", type=int, default=20, help="Number of local centers kept for each image.")
    parser.add_argument("--local_iters", type=int, default=10, help="KMeans iterations inside each image.")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--chunk_images", type=int, default=64, help="Number of images processed per chunk.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--compute_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_images", type=int, default=None, help="Debug only: use first N images across shards.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.local_k <= 0:
        raise ValueError("--local_k must be positive")
    if args.local_iters <= 0:
        raise ValueError("--local_iters must be positive")
    if args.chunk_images <= 0:
        raise ValueError("--chunk_images must be positive")

    set_seed(args.seed)
    build_local_centers(args)


if __name__ == "__main__":
    main()
