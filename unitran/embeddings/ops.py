"""Common tensor and embedding operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def safe_torch_load(path: str | Path) -> Any:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pick_tensor(obj: Any, key: str, path: str | Path) -> torch.Tensor:
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


def normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=1e-12)


def pad_to_same_dim(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim_a = tensor_a.size(1)
    dim_b = tensor_b.size(1)
    target_dim = max(dim_a, dim_b)

    def pad_tensor(t: torch.Tensor, curr_dim: int) -> torch.Tensor:
        if curr_dim == target_dim:
            return t
        padded = torch.zeros(t.size(0), target_dim, dtype=t.dtype, device=t.device)
        padded[:, :curr_dim] = t
        return padded

    return pad_tensor(tensor_a, dim_a), pad_tensor(tensor_b, dim_b)


# Backward-compatible name used by the original scripts.
pad_to_same_dim_v2 = pad_to_same_dim
