#!/usr/bin/env python3
"""
Streaming full-batch KMeans for DINOv2 COCO patch tokens.

Input expected from the previous feature extraction step:

feature/coco2014_dinov2_vitb14_448/
  images.jsonl
  meta.json
  patch_tokens_shape.json
  patch_tokens/
    shard_000_fp16.mmap
    shard_000.meta.json
    shard_001_fp16.mmap
    shard_001.meta.json
    ...

This script does NOT load all patch tokens into RAM. Each KMeans iteration streams
through all feature shards and updates centroids from every patch token.

Default target:
  DINOv2 ViT-B/14, 448x448 -> [N, 1024, 768]
  K=1024, cosine/spherical KMeans, fp16 storage/search, fp32 accumulation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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

    @property
    def num_vectors(self) -> int:
        return self.num_images * self.num_patches


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def append_jsonl(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
            # Prefer path relative to feature_dir; fallback to patch_dir/name.
            candidate = feature_dir / mmap_value
            mmap_path = candidate if candidate.exists() else patch_dir / Path(mmap_value).name
        if not mmap_path.exists():
            # Common layout from the extraction script.
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
                shape=shape,  # possibly truncated for debug
                dtype=dtype,
                global_indices=global_indices,
            )
        )

    if not shards:
        raise RuntimeError("No usable shards loaded.")

    num_patches = shards[0].num_patches
    dim = shards[0].dim
    for s in shards:
        if s.num_patches != num_patches or s.dim != dim:
            raise ValueError(f"Shard shape mismatch: {s.meta_path} shape={s.shape}, expected P={num_patches}, D={dim}")
    return shards


def open_mmap(shard: ShardInfo, mode: str = "r") -> np.memmap:
    np_dtype = np.float16 if shard.dtype == "float16" else np.float32
    return np.memmap(shard.mmap_path, dtype=np_dtype, mode=mode, shape=shard.shape)


def iter_image_chunks(shards: Sequence[ShardInfo], chunk_images: int) -> Iterator[Tuple[ShardInfo, int, int, np.ndarray]]:
    for shard in shards:
        mmap = open_mmap(shard, mode="r")
        for start in range(0, shard.num_images, chunk_images):
            end = min(start + chunk_images, shard.num_images)
            arr = np.asarray(mmap[start:end])
            yield shard, start, end, arr
        del mmap


def to_device_chunk(
    arr: np.ndarray,
    device: torch.device,
    metric: str,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    # arr: [B, P, D] fp16/fp32 memmap slice
    # Make a writable contiguous CPU buffer before torch.from_numpy.
    # This avoids read-only memmap warnings and makes H2D transfer predictable.
    x = np.array(arr.reshape(-1, arr.shape[-1]), copy=True, order="C")
    t = torch.from_numpy(x).to(device=device, non_blocking=True)
    if metric == "cosine":
        # normalize in fp32 for stability, then cast for search if requested
        t = F.normalize(t.float(), dim=1, eps=1e-12)
    else:
        t = t.float()
    if compute_dtype in (torch.float16, torch.bfloat16) and device.type == "cuda":
        t = t.to(compute_dtype)
    return t


def assign_chunk(
    x: torch.Tensor,
    centroids: torch.Tensor,
    metric: str,
    distance_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return labels and scores/distances for a chunk.

    For cosine: returns labels and distance = 1 - max cosine similarity.
    For l2: returns labels and squared L2 distance.
    """
    if distance_chunk and x.shape[0] > distance_chunk:
        labels_list = []
        dist_list = []
        for st in range(0, x.shape[0], distance_chunk):
            lab, dist = assign_chunk(x[st: st + distance_chunk], centroids, metric, distance_chunk=0)
            labels_list.append(lab)
            dist_list.append(dist)
        return torch.cat(labels_list, dim=0), torch.cat(dist_list, dim=0)

    if metric == "cosine":
        # Spherical KMeans: nearest centroid by max inner product of normalized vectors.
        sims = x @ centroids.T
        best_sim, labels = torch.max(sims, dim=1)
        dist = 1.0 - best_sim.float()
        return labels.to(torch.long), dist

    if metric == "l2":
        # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
        x_f = x.float()
        c_f = centroids.float()
        x2 = (x_f * x_f).sum(dim=1, keepdim=True)
        c2 = (c_f * c_f).sum(dim=1).view(1, -1)
        dist_mat = x2 + c2 - 2.0 * (x_f @ c_f.T)
        dist, labels = torch.min(dist_mat, dim=1)
        return labels.to(torch.long), dist

    raise ValueError(f"Unknown metric: {metric}")


def init_random_centroids(
    shards: Sequence[ShardInfo],
    k: int,
    device: torch.device,
    metric: str,
    compute_dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    total = sum(s.num_vectors for s in shards)
    if total < k:
        raise ValueError(f"Need at least k vectors, got total_vectors={total}, k={k}")

    # Random global positions over concatenated shard vectors.
    chosen = rng.choice(total, size=k, replace=False)
    chosen.sort()

    centroids_np = []
    offset = 0
    pos_i = 0
    for shard in shards:
        shard_total = shard.num_vectors
        local_positions = []
        while pos_i < len(chosen) and offset <= chosen[pos_i] < offset + shard_total:
            local_positions.append(int(chosen[pos_i] - offset))
            pos_i += 1
        if local_positions:
            mmap = open_mmap(shard, mode="r")
            flat = np.asarray(mmap).reshape(-1, shard.dim)
            centroids_np.append(np.ascontiguousarray(flat[local_positions]))
            del mmap
        offset += shard_total

    centroids = np.concatenate(centroids_np, axis=0).astype("float32", copy=False)
    c = torch.from_numpy(centroids).to(device=device)
    if metric == "cosine":
        c = F.normalize(c, dim=1, eps=1e-12)
    if compute_dtype in (torch.float16, torch.bfloat16) and device.type == "cuda":
        c = c.to(compute_dtype)
    return c.contiguous()


def maybe_load_centroids(path: Optional[str], device: torch.device, metric: str, compute_dtype: torch.dtype) -> Optional[torch.Tensor]:
    if not path:
        return None
    arr = np.load(path).astype("float32", copy=False)
    c = torch.from_numpy(arr).to(device=device)
    if metric == "cosine":
        c = F.normalize(c, dim=1, eps=1e-12)
    if compute_dtype in (torch.float16, torch.bfloat16) and device.type == "cuda":
        c = c.to(compute_dtype)
    return c.contiguous()


def save_centroids(kmeans_dir: Path, k: int, centroids: torch.Tensor) -> Tuple[Path, Path]:
    c32 = centroids.detach().float().cpu().numpy().astype("float32")
    c16 = c32.astype("float16")
    p32 = kmeans_dir / f"centroids_k{k}_fp32.npy"
    p16 = kmeans_dir / f"centroids_k{k}_fp16.npy"
    np.save(p32, c32)
    np.save(p16, c16)
    return p32, p16


def run_kmeans(args: argparse.Namespace, shards: Sequence[ShardInfo]) -> Path:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    compute_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.compute_dtype]

    feature_dir = Path(args.feature_dir)
    kmeans_dir = feature_dir / "kmeans"
    kmeans_dir.mkdir(parents=True, exist_ok=True)
    log_path = kmeans_dir / f"train_log_k{args.k}.jsonl"
    if args.overwrite and log_path.exists():
        log_path.unlink()

    centroids = maybe_load_centroids(args.init_centroids, device, args.metric, compute_dtype)
    if centroids is None:
        centroids = init_random_centroids(
            shards=shards,
            k=args.k,
            device=device,
            metric=args.metric,
            compute_dtype=compute_dtype,
            seed=args.seed,
        )
    if centroids.shape != (args.k, shards[0].dim):
        raise ValueError(f"Centroid shape mismatch: got {tuple(centroids.shape)}, expected {(args.k, shards[0].dim)}")

    config = {
        "feature_dir": str(feature_dir),
        "k": args.k,
        "num_iters": args.num_iters,
        "metric": args.metric,
        "algorithm": "streaming_full_batch_kmeans",
        "feature_dtype_on_disk": shards[0].dtype,
        "compute_dtype": args.compute_dtype,
        "accumulate_dtype": "float32",
        "chunk_images": args.chunk_images,
        "num_shards": len(shards),
        "num_images": int(sum(s.num_images for s in shards)),
        "num_patches_per_image": shards[0].num_patches,
        "feature_dim": shards[0].dim,
        "total_patch_tokens": int(sum(s.num_vectors for s in shards)),
        "seed": args.seed,
        "empty_cluster_policy": args.empty_cluster_policy,
        "device": str(device),
    }
    json_dump(kmeans_dir / f"config_k{args.k}.json", config)

    print(f"[kmeans] device={device}, dtype={args.compute_dtype}, metric={args.metric}")
    print(f"[kmeans] images={config['num_images']}, vectors={config['total_patch_tokens']:,}, dim={shards[0].dim}, k={args.k}")
    print(f"[kmeans] chunk_images={args.chunk_images}, per chunk vectors≈{args.chunk_images * shards[0].num_patches:,}")

    old_centroids_for_empty = centroids.detach().float().clone()

    for it in range(args.num_iters):
        t0 = time.time()
        sum_buf = torch.zeros((args.k, shards[0].dim), dtype=torch.float32, device=device)
        count_buf = torch.zeros((args.k,), dtype=torch.float32, device=device)
        total_loss = 0.0
        total_vectors = 0

        progress = tqdm(iter_image_chunks(shards, args.chunk_images), desc=f"KMeans iter {it+1}/{args.num_iters}")
        for shard, start, end, arr in progress:
            x = to_device_chunk(arr, device=device, metric=args.metric, compute_dtype=compute_dtype)
            labels, dist = assign_chunk(x, centroids, args.metric, args.distance_chunk)

            # Accumulate in fp32 even if search is fp16.
            x_acc = x.float()
            sum_buf.index_add_(0, labels, x_acc)
            ones = torch.ones_like(labels, dtype=torch.float32, device=device)
            count_buf.index_add_(0, labels, ones)

            total_loss += float(dist.float().sum().detach().cpu())
            total_vectors += int(labels.numel())
            progress.set_postfix(loss=f"{total_loss / max(total_vectors, 1):.6f}")

            del x, x_acc, labels, dist, ones

        non_empty = count_buf > 0
        new_centroids = sum_buf / count_buf.clamp_min(1.0).unsqueeze(1)
        num_empty = int((~non_empty).sum().detach().cpu())

        if num_empty > 0:
            if args.empty_cluster_policy == "keep_old":
                new_centroids[~non_empty] = old_centroids_for_empty[~non_empty].to(device)
            elif args.empty_cluster_policy == "random_reinit":
                reinit = init_random_centroids(
                    shards=shards,
                    k=num_empty,
                    device=device,
                    metric=args.metric,
                    compute_dtype=torch.float32,
                    seed=args.seed + 10000 + it,
                ).float()
                new_centroids[~non_empty] = reinit
            else:
                raise ValueError(f"Unknown empty_cluster_policy={args.empty_cluster_policy}")

        if args.metric == "cosine":
            new_centroids = F.normalize(new_centroids.float(), dim=1, eps=1e-12)

        shift = torch.norm(new_centroids - centroids.float(), dim=1).mean().item()
        old_centroids_for_empty = new_centroids.detach().clone()

        if compute_dtype in (torch.float16, torch.bfloat16) and device.type == "cuda":
            centroids = new_centroids.to(compute_dtype).contiguous()
        else:
            centroids = new_centroids.float().contiguous()

        count_cpu = count_buf.detach().cpu().numpy()
        log = {
            "iter": it + 1,
            "num_vectors": int(total_vectors),
            "avg_loss": float(total_loss / max(total_vectors, 1)),
            "centroid_shift_mean": float(shift),
            "num_empty_clusters": int(num_empty),
            "min_cluster_count": int(count_cpu.min()) if len(count_cpu) else 0,
            "max_cluster_count": int(count_cpu.max()) if len(count_cpu) else 0,
            "mean_cluster_count": float(count_cpu.mean()) if len(count_cpu) else 0.0,
            "elapsed_sec": round(time.time() - t0, 3),
        }
        append_jsonl(log_path, log)
        print(f"[iter {it+1}] loss={log['avg_loss']:.6f}, shift={shift:.6f}, empty={num_empty}, elapsed={log['elapsed_sec']}s")

        p32, p16 = save_centroids(kmeans_dir, args.k, centroids)
        np.save(kmeans_dir / f"cluster_counts_k{args.k}.npy", count_cpu.astype("int64"))

    print(f"[save] {p32}")
    print(f"[save] {p16}")
    print(f"[save] {log_path}")
    return p32


def run_assignment(args: argparse.Namespace, shards: Sequence[ShardInfo], centroid_path: Path) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    compute_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.compute_dtype]

    feature_dir = Path(args.feature_dir)
    assignment_dir = feature_dir / "assignment"
    assignment_dir.mkdir(parents=True, exist_ok=True)

    total_images = int(max(int(s.global_indices.max()) for s in shards) + 1)
    num_patches = shards[0].num_patches
    dim = shards[0].dim
    label_dtype = np.uint16 if args.k <= 65535 else np.uint32

    ids_path = assignment_dir / f"patch_cluster_ids_k{args.k}_{np.dtype(label_dtype).name}.npy"
    dist_path = assignment_dir / f"patch_cluster_dist_k{args.k}_fp16.npy"

    if ids_path.exists() and not args.overwrite:
        raise FileExistsError(f"{ids_path} exists. Use --overwrite to rewrite assignment.")
    if args.save_dist and dist_path.exists() and not args.overwrite:
        raise FileExistsError(f"{dist_path} exists. Use --overwrite to rewrite distances.")

    ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=label_dtype, shape=(total_images, num_patches))
    dists = None
    if args.save_dist:
        dists = np.lib.format.open_memmap(dist_path, mode="w+", dtype=np.float16, shape=(total_images, num_patches))

    centroids = maybe_load_centroids(str(centroid_path), device, args.metric, compute_dtype)
    assert centroids is not None
    if centroids.shape != (args.k, dim):
        raise ValueError(f"Centroid shape mismatch: got {tuple(centroids.shape)}, expected {(args.k, dim)}")

    print(f"[assign] centroid={centroid_path}")
    print(f"[assign] output ids={ids_path}, shape={(total_images, num_patches)}, dtype={label_dtype}")

    total_vectors = 0
    t0 = time.time()
    progress = tqdm(iter_image_chunks(shards, args.chunk_images), desc="assignment")
    for shard, start, end, arr in progress:
        x = to_device_chunk(arr, device=device, metric=args.metric, compute_dtype=compute_dtype)
        labels, dist = assign_chunk(x, centroids, args.metric, args.distance_chunk)

        b = end - start
        labels_np = labels.detach().cpu().numpy().reshape(b, num_patches).astype(label_dtype, copy=False)
        global_idx = shard.global_indices[start:end]
        ids[global_idx] = labels_np

        if dists is not None:
            dist_np = dist.detach().cpu().numpy().reshape(b, num_patches).astype("float16", copy=False)
            dists[global_idx] = dist_np

        total_vectors += int(labels.numel())
        progress.set_postfix(vectors=f"{total_vectors:,}")
        del x, labels, dist

    ids.flush()
    if dists is not None:
        dists.flush()

    config = {
        "k": args.k,
        "metric": args.metric,
        "centroid_path": str(centroid_path),
        "ids_path": str(ids_path),
        "dist_path": str(dist_path) if args.save_dist else None,
        "shape": [total_images, num_patches],
        "id_dtype": np.dtype(label_dtype).name,
        "distance_dtype": "float16" if args.save_dist else None,
        "chunk_images": args.chunk_images,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    json_dump(assignment_dir / f"config_assignment_k{args.k}.json", config)
    print(f"[done] assignment vectors={total_vectors:,}, elapsed={config['elapsed_sec']}s")
    print(f"[save] {ids_path}")
    if args.save_dist:
        print(f"[save] {dist_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming full-batch KMeans for COCO DINOv2 patch tokens.")
    parser.add_argument("--feature_dir", type=str, default="feature/coco2014_dinov2_vitb14_448")
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--chunk_images", type=int, default=128, help="Number of images per streaming chunk.")
    parser.add_argument("--distance_chunk", type=int, default=0, help="Optional vector subchunk for distance matrix. 0 disables.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--compute_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--init_centroids", type=str, default=None, help="Optional .npy centroids to continue/resume training.")
    parser.add_argument("--empty_cluster_policy", type=str, default="keep_old", choices=["keep_old", "random_reinit"])
    parser.add_argument("--max_images", type=int, default=None, help="Debug only: use first N images across shards.")
    parser.add_argument("--assign", action="store_true", help="Run full assignment after KMeans training.")
    parser.add_argument("--only_assign", action="store_true", help="Skip training and only assign using --init_centroids or existing centroids file.")
    parser.add_argument("--save_dist", action="store_true", help="Also save nearest distance as fp16 .npy.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    feature_dir = Path(args.feature_dir)
    shards = load_shards(feature_dir, max_images=args.max_images)

    print("[data] shards:")
    for s in shards:
        print(f"  shard_{s.shard_id:03d}: images={s.num_images}, shape={s.shape}, dtype={s.dtype}, mmap={s.mmap_path}")

    if args.only_assign:
        centroid_path = Path(args.init_centroids) if args.init_centroids else feature_dir / "kmeans" / f"centroids_k{args.k}_fp32.npy"
        if not centroid_path.exists():
            raise FileNotFoundError(f"Centroid file not found: {centroid_path}")
    else:
        centroid_path = run_kmeans(args, shards)

    if args.assign or args.only_assign:
        run_assignment(args, shards, centroid_path)


if __name__ == "__main__":
    main()
