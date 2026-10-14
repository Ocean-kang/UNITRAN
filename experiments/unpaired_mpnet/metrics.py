"""Exact blockwise paired retrieval metrics on the independent COCO validation set."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def retrieval(query: torch.Tensor, keys: torch.Tensor, block_size: int = 128,
              device: str = "cpu") -> dict:
    """Aligned rows are ground-truth matches; arbitrary row-order on either side is invalid.

    Midrank accounts for tied similarities. Full NxN similarity is never materialized.
    """
    if query.ndim != 2 or keys.ndim != 2 or query.shape != keys.shape:
        raise ValueError("Expected same-shaped [N,D] paired embeddings")
    if len(query) < 2:
        raise ValueError("Retrieval requires at least 2 validation pairs")
    dev = torch.device(device)
    a = F.normalize(query.float(), dim=-1).to(dev)
    b = F.normalize(keys.float(), dim=-1).to(dev)
    n = len(a)
    ranks = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        scores = a[start:end] @ b.T
        true = scores[torch.arange(end - start, device=dev), torch.arange(start, end, device=dev)]
        # The true partner cannot count as an additional tie. Same rule as Rosetta midranks.
        rank = (scores > true[:, None]).sum(dim=1).double() + 1
        tied = (scores == true[:, None]).sum(dim=1).double() - 1
        ranks.append((rank + .5 * tied).cpu())
    rank = torch.cat(ranks)
    return {
        "num_validation_samples": n,
        "foscttm": ((rank - 1) / (n - 1)).mean().item(),
        "mean_rank": rank.mean().item(),
        "top1": (rank <= 1).double().mean().item(),
        "r_at_1": (rank <= 1).double().mean().item(),
        "r_at_5": (rank <= 5).double().mean().item(),
        "r_at_10": (rank <= 10).double().mean().item(),
    }


def topk_cosine_indices(a: torch.Tensor, b: torch.Tensor, k: int,
                        block_size: int, device: str = "cpu") -> torch.Tensor:
    """Like UNITRAN's cos_sim_matrix(a,b).topk(k), without allocating its NxM matrix."""
    if not 1 <= k <= len(b):
        raise ValueError(f"top-k={k} exceeds target samples {len(b)}")
    dev = torch.device(device)
    b_norm = F.normalize(b.float(), dim=-1).to(dev)
    indices = []
    for start in range(0, len(a), block_size):
        x = F.normalize(a[start:start + block_size].float(), dim=-1).to(dev)
        idx = (x @ b_norm.T).topk(k, dim=-1).indices.cpu()
        indices.append(idx)
    return torch.cat(indices, dim=0)


def mean_selected(bank: torch.Tensor, indices: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Produce the original UNITRAN top-k mean without large [N,K,D] intermediates."""
    out = []
    for st in range(0, len(indices), batch_size):
        out.append(bank[indices[st:st + batch_size]].mean(dim=1))
    return torch.cat(out, dim=0)
