"""Reusable distribution metrics for vision/text embedding comparison."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    outs = []
    y2 = (y * y).sum(dim=1)[None, :]
    for i in range(0, x.size(0), chunk):
        xb = x[i : i + chunk]
        d = (xb * xb).sum(dim=1)[:, None] + y2 - 2 * xb @ y.T
        outs.append(d.clamp_min_(0).cpu())
    return torch.cat(outs, dim=0)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float | None = None) -> float:
    x = x.float().cpu()
    y = y.float().cpu()
    if sigma is None:
        z = torch.cat([x, y], dim=0)
        d = pairwise_sq_dists(z, z)
        vals = d[d > 0]
        sigma = float(torch.sqrt(torch.median(vals)).item()) if vals.numel() else 1.0
    gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)
    kxx = torch.exp(-gamma * pairwise_sq_dists(x, x)).mean()
    kyy = torch.exp(-gamma * pairwise_sq_dists(y, y)).mean()
    kxy = torch.exp(-gamma * pairwise_sq_dists(x, y)).mean()
    return float((kxx + kyy - 2 * kxy).item())


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, num_proj: int = 256, seed: int = 0) -> float:
    x = x.float().cpu()
    y = y.float().cpu()
    n = min(len(x), len(y))
    g = torch.Generator().manual_seed(seed)
    x = x[torch.randperm(len(x), generator=g)[:n]]
    y = y[torch.randperm(len(y), generator=g)[:n]]
    proj = torch.randn(x.size(1), num_proj, generator=g)
    proj = F.normalize(proj, dim=0, eps=1e-12)
    xs = torch.sort(x @ proj, dim=0).values
    ys = torch.sort(y @ proj, dim=0).values
    return float((xs - ys).abs().mean().item())


def kth_radius(x: torch.Tensor, refs: torch.Tensor, k: int) -> torch.Tensor:
    d = torch.cdist(x.float(), refs.float())
    return d.kthvalue(k=min(k, d.size(1)), dim=1).values


def precision_recall_coverage(x: torch.Tensor, y: torch.Tensor, k: int = 5) -> Tuple[float, float]:
    x = x.float().cpu()
    y = y.float().cpu()
    rx = kth_radius(x, x, k + 1)
    ry = kth_radius(y, y, k + 1)
    dxy = torch.cdist(x, y)
    precision = (dxy <= ry[None, :]).any(dim=1).float().mean().item()
    recall = (dxy <= rx[:, None]).any(dim=0).float().mean().item()
    return float(precision), float(recall)


def pca_participation_ratio(x: torch.Tensor) -> float:
    x = x.float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    eig = s.pow(2)
    return float((eig.sum().pow(2) / (eig.pow(2).sum() + 1e-12)).item())
