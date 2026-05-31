import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize(x, dim=-1):
    return F.normalize(x, dim=dim)


def pad_to_same_dim_v2(tensor_a, tensor_b):
    dim_a = tensor_a.size(1)
    dim_b = tensor_b.size(1)
    target_dim = max(dim_a, dim_b)

    def pad_tensor(t, curr_dim):
        if curr_dim == target_dim:
            return t
        padded = torch.zeros(t.size(0), target_dim, dtype=t.dtype, device=t.device)
        padded[:, :curr_dim] = t
        return padded

    return pad_tensor(tensor_a, dim_a), pad_tensor(tensor_b, dim_b)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pick_tensor(obj, key, path):
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


def as_2d(x, name):
    x = x.float().cpu()
    if x.ndim == 3 and x.size(1) == 1:
        x = x[:, 0]
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape (N, D), but got {tuple(x.shape)}")
    return x


def load_pair(args):
    if args.input_pt is not None:
        obj = safe_torch_load(Path(args.input_pt))
        vision = pick_tensor(obj, args.vision_key, args.input_pt)
        text = pick_tensor(obj, args.text_key, args.input_pt)
    else:
        if args.vision_pt is None or args.text_pt is None:
            raise ValueError("Use either --input_pt, or both --vision_pt and --text_pt")
        vision_obj = safe_torch_load(Path(args.vision_pt))
        text_obj = safe_torch_load(Path(args.text_pt))
        vision = pick_tensor(vision_obj, args.vision_key, args.vision_pt)
        text = pick_tensor(text_obj, args.text_key, args.text_pt)

    return as_2d(vision, "vision"), as_2d(text, "text")


def sample_points(x, max_points, seed):
    if max_points is None or max_points <= 0 or len(x) <= max_points:
        return x
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=rng)[:max_points]
    return x[idx]


def sample_pair(vision, text, max_points, seed):
    if len(vision) == len(text):
        if max_points is None or max_points <= 0 or len(vision) <= max_points:
            return vision, text, True
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(vision), generator=rng)[:max_points]
        return vision[idx], text[idx], True
    return sample_points(vision, max_points, seed), sample_points(text, max_points, seed + 1), False


def preprocess_pair(vision, text, mode):
    vision, text = pad_to_same_dim_v2(vision, text)

    if mode == "raw":
        return vision, text
    if mode == "l2":
        return normalize(vision), normalize(text)
    if mode == "unitran":
        vision = normalize(vision - vision.mean(dim=0))
        text = normalize(text - text.mean(dim=0))
        return vision, text

    raise ValueError(f"Unknown preprocess mode: {mode}")


def cosine_mean_metrics(vision, text, chunk_size, same_index_available):
    text_t = text.T
    total = 0.0
    count = 0
    for i in range(0, len(vision), chunk_size):
        sim = vision[i:i + chunk_size] @ text_t
        total += sim.sum().item()
        count += sim.numel()

    out = {"all_pair_cosine_mean": total / count}
    if same_index_available and len(vision) == len(text):
        out["same_index_cosine_mean"] = F.cosine_similarity(vision, text, dim=-1).mean().item()
    return out


def pairwise_sq_dists(x, y):
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return (x2 + y2 - 2 * x @ y.T).clamp_min(0)


def mmd_rbf(vision, text, sigma=None):
    xx = pairwise_sq_dists(vision, vision)
    yy = pairwise_sq_dists(text, text)
    xy = pairwise_sq_dists(vision, text)

    if sigma is None:
        sample = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])
        sample = sample[sample > 0]
        sigma = torch.sqrt(torch.median(sample) / 2).item() if sample.numel() > 0 else 1.0
        sigma = max(sigma, 1e-6)

    gamma = 1.0 / (2 * sigma * sigma)
    mmd2 = torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean() - 2 * torch.exp(-gamma * xy).mean()
    return {"mmd_rbf": float(mmd2.item()), "mmd_sigma": float(sigma)}


def cluster_wasserstein(vision, text, k, seed):
    k = min(k, len(vision), len(text))
    if k < 2:
        raise ValueError("--k must be at least 2 after considering sample counts")

    km_v = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(vision.numpy())
    km_t = KMeans(n_clusters=k, n_init=10, random_state=seed + 1).fit(text.numpy())
    centers_v = torch.tensor(km_v.cluster_centers_).float()
    centers_t = torch.tensor(km_t.cluster_centers_).float()
    cost = torch.cdist(centers_v, centers_t, p=2).numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return {"kmeans_wasserstein": float(cost[row_ind, col_ind].mean()), "kmeans_k": int(k)}


def cosine_distances(x, y):
    return (1 - x @ y.T).clamp_min(0)


def kth_radius(x, k):
    dist = cosine_distances(x, x)
    dist.fill_diagonal_(float("inf"))
    k = min(k, len(x) - 1)
    return dist.kthvalue(k, dim=1).values


def precision_recall_coverage(vision, text, k):
    if len(vision) < 2 or len(text) < 2:
        raise ValueError("precision/recall coverage needs at least 2 points in each space")

    r_v = kth_radius(vision, k)
    r_t = kth_radius(text, k)
    dist_vt = cosine_distances(vision, text)

    nn_v_for_t_dist, nn_v_for_t = dist_vt.min(dim=0)
    nn_t_for_v_dist, nn_t_for_v = dist_vt.min(dim=1)

    precision_t_in_v = (nn_v_for_t_dist <= r_v[nn_v_for_t]).float().mean().item()
    recall_v_in_t = (nn_t_for_v_dist <= r_t[nn_t_for_v]).float().mean().item()
    denom = precision_t_in_v + recall_v_in_t
    f1 = 0.0 if denom == 0 else 2 * precision_t_in_v * recall_v_in_t / denom

    coverage_v_by_t = nn_v_for_t.unique().numel() / len(vision)
    coverage_t_by_v = nn_t_for_v.unique().numel() / len(text)

    return {
        "precision_text_in_vision": float(precision_t_in_v),
        "recall_vision_in_text": float(recall_v_in_t),
        "precision_recall_f1": float(f1),
        "coverage_vision_by_text": float(coverage_v_by_t),
        "coverage_text_by_vision": float(coverage_t_by_v),
        "coverage_ratio": float(0.5 * (coverage_v_by_t + coverage_t_by_v)),
        "coverage_knn_k": int(k),
    }


def pairwise_spearman(vision, text):
    n = min(len(vision), len(text))
    vision = vision[:n]
    text = text[:n]

    sim_v = vision @ vision.T
    sim_t = text @ text.T
    tri = torch.triu_indices(n, n, offset=1)
    a = sim_v[tri[0], tri[1]].numpy()
    b = sim_t[tri[0], tri[1]].numpy()
    corr = spearmanr(a, b).correlation
    if np.isnan(corr):
        corr = 0.0
    return {"pairwise_spearman": float(corr), "pairwise_spearman_n": int(n)}


def sinkhorn_uniform(cost, epsilon, iters):
    n, m = cost.shape
    log_k = -cost / epsilon
    log_u = torch.zeros(n)
    log_v = torch.zeros(m)
    log_p = torch.full((n,), -np.log(n))
    log_q = torch.full((m,), -np.log(m))

    for _ in range(iters):
        log_u = log_p - torch.logsumexp(log_k + log_v[None, :], dim=1)
        log_v = log_q - torch.logsumexp(log_k + log_u[:, None], dim=0)

    return torch.exp(log_k + log_u[:, None] + log_v[None, :])


def entropic_gw_distance(vision, text, epsilon, gw_iters, sinkhorn_iters):
    c_v = cosine_distances(vision, vision)
    c_t = cosine_distances(text, text)
    c_v = c_v / c_v.max().clamp_min(1e-6)
    c_t = c_t / c_t.max().clamp_min(1e-6)

    n, m = len(vision), len(text)
    p = torch.full((n,), 1.0 / n)
    q = torch.full((m,), 1.0 / m)
    t = torch.outer(p, q)

    c_v2_p = (c_v * c_v) @ p
    c_t2_q = (c_t * c_t) @ q
    const = c_v2_p[:, None] + c_t2_q[None, :]

    for _ in range(gw_iters):
        cost = const - 2 * c_v @ t @ c_t.T
        t = sinkhorn_uniform(cost, epsilon, sinkhorn_iters)

    cost = const - 2 * c_v @ t @ c_t.T
    return {"gw_distance": float((cost * t).sum().item()), "gw_epsilon": float(epsilon)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", type=str, default=None)
    parser.add_argument("--vision_pt", type=str, default=None)
    parser.add_argument("--text_pt", type=str, default=None)
    parser.add_argument("--vision_key", type=str, default="vision_feats")
    parser.add_argument("--text_key", type=str, default="text_feats")
    parser.add_argument("--preprocess", type=str, choices=["raw", "l2", "unitran"], default="unitran")
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--mmd_points", type=int, default=2000)
    parser.add_argument("--spearman_points", type=int, default=1000)
    parser.add_argument("--gw_points", type=int, default=300)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--coverage_k", type=int, default=5)
    parser.add_argument("--gw_epsilon", type=float, default=0.05)
    parser.add_argument("--gw_iters", type=int, default=30)
    parser.add_argument("--sinkhorn_iters", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_json", type=str, default="outputs/distribution_metric.json")
    args = parser.parse_args()

    set_seed(args.seed)

    vision_raw, text_raw = load_pair(args)
    raw_shapes = {"vision": list(vision_raw.shape), "text": list(text_raw.shape)}
    vision, text, same_index_available = sample_pair(vision_raw, text_raw, args.max_points, args.seed)
    vision, text = preprocess_pair(vision, text, args.preprocess)

    print(f"[data] raw vision={tuple(vision_raw.shape)}, raw text={tuple(text_raw.shape)}")
    print(f"[data] sampled vision={tuple(vision.shape)}, sampled text={tuple(text.shape)}, preprocess={args.preprocess}")

    results = {
        "input_pt": args.input_pt,
        "vision_pt": args.vision_pt,
        "text_pt": args.text_pt,
        "vision_key": args.vision_key,
        "text_key": args.text_key,
        "preprocess": args.preprocess,
        "raw_shapes": raw_shapes,
        "sampled_shapes": {"vision": list(vision.shape), "text": list(text.shape)},
        "same_index_available": bool(same_index_available),
        "seed": args.seed,
    }

    results.update(cosine_mean_metrics(vision, text, args.chunk_size, same_index_available))

    mmd_v = sample_points(vision, args.mmd_points, args.seed + 10)
    mmd_t = sample_points(text, args.mmd_points, args.seed + 11)
    results.update(mmd_rbf(mmd_v, mmd_t))

    results.update(cluster_wasserstein(vision, text, args.k, args.seed))
    results.update(precision_recall_coverage(vision, text, args.coverage_k))

    sp_n = min(len(vision), len(text), args.spearman_points)
    results.update(pairwise_spearman(vision[:sp_n], text[:sp_n]))

    gw_v = sample_points(vision, args.gw_points, args.seed + 30)
    gw_t = sample_points(text, args.gw_points, args.seed + 31)
    results.update(entropic_gw_distance(gw_v, gw_t, args.gw_epsilon, args.gw_iters, args.sinkhorn_iters))
    results["gw_points_used"] = {"vision": len(gw_v), "text": len(gw_t)}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
