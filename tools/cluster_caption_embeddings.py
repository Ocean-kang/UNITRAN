#!/usr/bin/env python3
"""KMeans clustering for caption/text embeddings saved as .pt or .npy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_load(path: Path) -> Any:
    if path.suffix == ".npy":
        return np.load(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pick_tensor(obj: Any, key: str, path: Path) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        if key not in obj:
            raise KeyError(f"{path} does not contain key '{key}'. Available keys: {list(obj.keys())}")
        value = obj[key]
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{path}['{key}'] is not a tensor or numpy array")
        return value
    raise TypeError(f"{path} must be a tensor, numpy array, or dict of tensors")


def as_2d(x: torch.Tensor, name: str) -> torch.Tensor:
    x = x.float().cpu()
    if x.ndim == 3 and x.size(1) == 1:
        x = x[:, 0]
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape (N, D), but got {tuple(x.shape)}")
    return x


def preprocess(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return x
    if mode == "l2":
        return F.normalize(x, dim=1, eps=1e-12)
    if mode == "unitran":
        return F.normalize(x - x.mean(dim=0), dim=1, eps=1e-12)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="KMeans clustering for caption/text embeddings.")
    parser.add_argument("--input_pt", type=str, required=True, help=".pt/.npy containing caption/text embeddings.")
    parser.add_argument("--text_key", type=str, default="text_feats")
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--preprocess", type=str, default="l2", choices=["raw", "l2", "unitran"])
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="feature/caption_kmeans")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    in_path = Path(args.input_pt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    centers_fp32_path = out_dir / f"centroids_k{args.k}_fp32.npy"
    if centers_fp32_path.exists() and not args.overwrite:
        raise FileExistsError(f"{centers_fp32_path} exists. Use --overwrite to rewrite outputs.")

    obj = safe_load(in_path)
    text = as_2d(pick_tensor(obj, args.text_key, in_path), "text")
    if len(text) < args.k:
        raise ValueError(f"Need at least k embeddings, got N={len(text)}, k={args.k}")
    text = preprocess(text, args.preprocess)

    print(f"[data] text={tuple(text.shape)}, preprocess={args.preprocess}")
    print(f"[kmeans] k={args.k}, n_init={args.n_init}, max_iter={args.max_iter}, seed={args.seed}")
    km = KMeans(n_clusters=args.k, n_init=args.n_init, max_iter=args.max_iter, random_state=args.seed)
    labels = km.fit_predict(text.numpy())
    centers = km.cluster_centers_.astype("float32", copy=False)
    if args.preprocess in ("l2", "unitran"):
        centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)

    counts = np.bincount(labels, minlength=args.k).astype("int64")
    np.save(centers_fp32_path, centers)
    np.save(out_dir / f"centroids_k{args.k}_fp16.npy", centers.astype("float16"))
    np.save(out_dir / f"cluster_ids_k{args.k}_int32.npy", labels.astype("int32"))
    np.save(out_dir / f"cluster_counts_k{args.k}.npy", counts)

    config = {
        "input_pt": str(in_path),
        "text_key": args.text_key,
        "k": args.k,
        "preprocess": args.preprocess,
        "algorithm": "sklearn_kmeans_on_caption_embeddings",
        "num_embeddings": int(text.shape[0]),
        "feature_dim": int(text.shape[1]),
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "out_dir": str(out_dir),
        "centroids_fp32": str(centers_fp32_path),
        "centroids_fp16": str(out_dir / f"centroids_k{args.k}_fp16.npy"),
        "cluster_ids": str(out_dir / f"cluster_ids_k{args.k}_int32.npy"),
        "cluster_counts": str(out_dir / f"cluster_counts_k{args.k}.npy"),
    }
    write_json(out_dir / f"config_k{args.k}.json", config)

    print(f"[done] inertia={km.inertia_:.6f}, empty_clusters={int((counts == 0).sum())}")
    print(f"[save] {centers_fp32_path}")
    print(f"[save] {out_dir / f'centroids_k{args.k}_fp16.npy'}")
    print(f"[save] {out_dir / f'cluster_ids_k{args.k}_int32.npy'}")
    print(f"[save] {out_dir / f'cluster_counts_k{args.k}.npy'}")
    print(f"[save] {out_dir / f'config_k{args.k}.json'}")


if __name__ == "__main__":
    main()
