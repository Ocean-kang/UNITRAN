"""mini-vec2vec core: no identities, captions, or evaluation tensors in fit()."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans

from src.evaluation.retrieval import mean_selected, topk_cosine_indices


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32).clone()


def cos_sim_matrix(x, y, device="cpu", chunk_size=None):
    x, y = tensor(x).to(device), tensor(y).to(device)
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    block = chunk_size or len(x)
    return torch.cat([(x[start:start + block] @ y.T).cpu() for start in range(0, len(x), block)])


def sim(x, y):
    x, y = tensor(x), tensor(y)
    return (x - x.mean(dim=0)) @ (y - y.mean(dim=0)).T


def train_orthogonal_linear(x, y):
    solution, _ = orthogonal_procrustes(np.asarray(x), np.asarray(y))
    return tensor(solution)


def eval_cos_score(x, y, weight):
    return F.cosine_similarity(x @ weight, y, dim=-1).mean().item()


def aligned_centroids(x, y, n_qap_runs=30, n_clusters=20, method="2opt", subsample=10000, seed=0):
    if method != "2opt":
        raise ValueError("mini-vec2vec uses QAP 2opt")
    g = torch.Generator().manual_seed(seed)
    if subsample is not None:
        x = x[torch.randperm(len(x), generator=g)[:subsample]]
        y = y[torch.randperm(len(y), generator=g)[:subsample]]
    a = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(x.numpy()).cluster_centers_
    b = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed + 1).fit(y.numpy()).cluster_centers_
    ka, kb = sim(a, a).double().numpy(), sim(b, b).double().numpy()
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_qap_runs):
        guess = np.column_stack((np.arange(n_clusters), rng.permutation(n_clusters)))
        result = quadratic_assignment(ka, kb, method="2opt",
                                      options={"maximize": True, "partial_guess": guess, "rng": rng})
        # The objective is maximized. The legacy implementation selected its minimum.
        if best is None or result.fun > best.fun:
            best = result
    return tensor(a), tensor(b[best.col_ind])


def preprocess(x, mean, dim):
    x = x.float().cpu()
    if x.ndim != 2 or x.shape[1] != mean.numel():
        raise ValueError("Encoder/checkpoint dimension mismatch")
    return F.pad(F.normalize(x - mean, dim=-1), (0, dim - x.shape[1]))


@torch.inference_mode()
def fit_unpaired(x, y, params):
    x, y = x.detach().float().cpu(), y.detach().float().cpu()
    for bank in (x, y):
        if bank.ndim != 2 or not torch.isfinite(bank).all():
            raise ValueError("Expected finite [N,D] training banks")
        if len(bank) < 2 or (bank - bank.mean(dim=0)).norm() < 1e-12:
            raise ValueError("Degenerate training distribution")
    p = params
    for key in ("num_anchor_runs", "num_clusters", "qap_runs", "anchor_subsample",
                "initial_k", "refine_sample", "refine_k", "refine2_clusters", "similarity_batch_size"):
        if int(p[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= p["alpha"] <= 1 or int(p["refine_iters"]) < 0:
        raise ValueError("Invalid smoothing/refinement count")
    if p["num_clusters"] > min(len(x), len(y), p["anchor_subsample"]) or p["refine2_clusters"] > min(len(x), len(y)):
        raise ValueError("Too few samples for configured cluster counts")
    if max(p["initial_k"], p["refine_k"]) > len(y):
        raise ValueError("Too few target samples for configured top-k")
    seed = int(p["seed"])
    mean_x, mean_y = x.mean(dim=0), y.mean(dim=0)
    dim = max(x.shape[1], y.shape[1])
    x, y = preprocess(x, mean_x, dim), preprocess(y, mean_y, dim)
    g = torch.Generator().manual_seed(seed)
    # Each side gets its own draw; no common row selection or hidden half-split.
    x, y = x[torch.randperm(len(x), generator=g)], y[torch.randperm(len(y), generator=g)]
    ax, ay = [], []
    for run in range(p["num_anchor_runs"]):
        a, b = aligned_centroids(x, y, p["qap_runs"], p["num_clusters"],
                                  subsample=p["anchor_subsample"], seed=seed + run)
        ax.append(a)
        ay.append(b)
    ax, ay = torch.cat(ax), torch.cat(ay)
    block, device = p["similarity_batch_size"], p["similarity_device"]
    rel_x, rel_y = cos_sim_matrix(x, ax), cos_sim_matrix(y, ay)
    indices = topk_cosine_indices(rel_x, rel_y, p["initial_k"], block, device)
    weight = train_orthogonal_linear(x, mean_selected(y, indices, block))
    del rel_x, rel_y, indices
    shifts = []
    for _ in range(p["refine_iters"]):
        sample = x[torch.randperm(len(x), generator=g)[:p["refine_sample"]]]
        indices = topk_cosine_indices(sample @ weight, y, p["refine_k"], block, device)
        new = train_orthogonal_linear(sample, mean_selected(y, indices, block))
        shifts.append((new - weight).norm().item())
        weight = (1 - p["alpha"]) * weight + p["alpha"] * new
    centers_x = tensor(KMeans(n_clusters=p["refine2_clusters"], n_init=10, random_state=seed).fit(x.numpy()).cluster_centers_)
    centers_y = tensor(KMeans(n_clusters=p["refine2_clusters"], init=(centers_x @ weight).numpy(),
                             n_init=1, random_state=seed).fit(y.numpy()).cluster_centers_)
    new = train_orthogonal_linear(centers_x, centers_y)
    weight = (1 - p["alpha"]) * weight + p["alpha"] * new
    return {"weight": weight, "mean_source": mean_x, "mean_target": mean_y, "dim": dim}, {
        "train_source_rows": len(x), "train_target_rows": len(y), "refinement_shifts": shifts,
        "algorithm": "KMeans/QAP -> relative-neighbor Procrustes -> NN refinement -> cluster correction",
    }
