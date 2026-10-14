"""Exact cosine retrieval with evaluator-only identities and bounded score memory."""
import torch
import torch.nn.functional as F


def topk_cosine_indices(a, b, k, block_size=128, device="cpu"):
    if block_size < 1 or not 1 <= k <= len(b):
        raise ValueError("Invalid block size or number of neighbors")
    target = F.normalize(b.float(), dim=-1).to(device)
    indices = []
    for start in range(0, len(a), block_size):
        query = F.normalize(a[start:start + block_size].float(), dim=-1).to(device)
        indices.append((query @ target.T).topk(k, dim=-1).indices.cpu())
    return torch.cat(indices)


def mean_selected(bank, indices, batch_size=128):
    return torch.cat([bank[indices[start:start + batch_size]].mean(dim=1)
                      for start in range(0, len(indices), batch_size)])


@torch.inference_mode()
def retrieval(query, keys, query_ids, key_ids, block_size=128, device="cpu", seed=123):
    """Midrank ties; R@K uses midrank <= K. Both modalities shuffled independently."""
    if block_size < 1 or query.ndim != 2 or keys.ndim != 2 or query.shape[1] != keys.shape[1]:
        raise ValueError("Invalid embeddings/block size")
    if len(query) != len(query_ids) or len(keys) != len(key_ids) or len(query) < 2:
        raise ValueError("Invalid identity counts")
    if len(set(query_ids)) != len(query_ids) or len(set(key_ids)) != len(key_ids) or set(query_ids) != set(key_ids):
        raise ValueError("Retrieval requires unique identical identity sets")
    if not torch.isfinite(query).all() or not torch.isfinite(keys).all():
        raise ValueError("Nonfinite retrieval vectors")
    g = torch.Generator().manual_seed(seed)
    ix, iy = torch.randperm(len(query), generator=g), torch.randperm(len(keys), generator=g)
    lookup = {key_ids[i]: pos for pos, i in enumerate(iy.tolist())}
    partner = torch.tensor([lookup[query_ids[i]] for i in ix.tolist()], device=device)
    a = F.normalize(query[ix].float(), dim=-1).to(device)
    b = F.normalize(keys[iy].float(), dim=-1).to(device)
    ranks = []
    for start in range(0, len(a), block_size):
        stop = min(start + block_size, len(a))
        scores = a[start:stop] @ b.T
        true = scores[torch.arange(stop - start, device=device), partner[start:stop]]
        greater = (scores > true[:, None]).sum(dim=1).double()
        ties = (scores == true[:, None]).sum(dim=1).double() - 1
        ranks.append((greater + 1 + .5 * ties).cpu())
    rank = torch.cat(ranks)
    return {"num_samples": len(a), "foscttm": ((rank - 1) / (len(b) - 1)).mean().item(),
            "mean_rank": rank.mean().item(),
            **{f"r_at_{k}": (rank <= k).double().mean().item() for k in (1, 5, 10)},
            "tie_policy": "midrank"}
