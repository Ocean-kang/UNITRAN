import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import OptimizeWarning, quadratic_assignment
from sklearn.cluster import KMeans
from tqdm.auto import trange

warnings.filterwarnings("ignore", category=OptimizeWarning)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor(x):
    return torch.tensor(x).float()


def normalize(x, dim=-1):
    return F.normalize(x, dim=dim)

def pad_to_same_dim_v2(tensor_a, tensor_b):
    dim_a = tensor_a.size(1)
    dim_b = tensor_b.size(1)
    target_dim = max(dim_a, dim_b)
    
    def pad_tensor(t, curr_dim):
        if curr_dim == target_dim:
            return t
        # 创建全零张量，形状 (batch, target_dim)
        padded = torch.zeros(t.size(0), target_dim, dtype=t.dtype, device=t.device)
        # 将原始数据填入前 curr_dim 列
        padded[:, :curr_dim] = t
        return padded
    
    return pad_tensor(tensor_a, dim_a), pad_tensor(tensor_b, dim_b)

def cos_sim_matrix(X, Y, device="cpu", chunk_size=None):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)

    X = X.to(device)
    Y = Y.to(device)

    X = X / X.norm(dim=-1, keepdim=True)
    Y = Y / Y.norm(dim=-1, keepdim=True)

    if chunk_size is None:
        return (X @ Y.T).cpu()

    outs = []
    for i in range(0, len(X), chunk_size):
        outs.append((X[i:i + chunk_size] @ Y.T).cpu())
    return torch.cat(outs, dim=0)


def sim(X, Y):
    X, Y = tensor(X), tensor(Y)
    H = torch.eye(len(X), device=X.device) - (1 / len(X)) * torch.ones((len(X), len(X)), device=X.device)
    return H @ X @ Y.T @ H


def train_orthogonal_linear(X, Y):
    solution, _ = orthogonal_procrustes(X, Y)
    return tensor(solution)


def eval_cos_score(X_eval, Y_eval, W):
    return torch.cosine_similarity(X_eval @ W, Y_eval, dim=-1).mean().item()


def aligned_centroids(
    X_train,
    Y_train,
    n_qap_runs=30,
    n_clusters=20,
    method="2opt",
    subsample=10_000,
    seed=0,
):
    options = {"P0": "randomized", "maximize": True}

    if subsample is not None:
        X_train = X_train[torch.randperm(len(X_train))[:subsample]]
        Y_train = Y_train[torch.randperm(len(Y_train))[:subsample]]

    clusterer1 = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    clusterer1.fit(X_train)

    clusterer2 = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed + 1)
    clusterer2.fit(Y_train)

    centers1, centers2 = clusterer1.cluster_centers_, clusterer2.cluster_centers_

    kernel1 = sim(centers1, centers1).float()
    kernel2 = sim(centers2, centers2).float()

    best_quad = None
    for _ in range(n_qap_runs):
        new_quad = quadratic_assignment(kernel1, kernel2, method=method, options=options)
        if best_quad is None or best_quad.fun < new_quad.fun:
            best_quad = new_quad

    centers2 = centers2[best_quad.col_ind]
    return tensor(centers1), tensor(centers2)


def compute_final_metrics(X_eval, Y_eval, W, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    X = (X_eval.to(device) @ W.to(device))
    Y = Y_eval.to(device)

    sim_mat = cos_sim_matrix(X, Y, device=device)
    ranks = torch.argsort(torch.argsort(sim_mat, dim=-1), dim=-1).diagonal()

    top1 = (len(X_eval) - 1 == ranks).float().mean().item()
    avg_rank = (len(X_eval) - ranks.float().mean()).item()
    return top1, avg_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", type=str, required=True)
    parser.add_argument("--source", type=str, default="clip")
    parser.add_argument("--target", type=str, default="dinov2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_anchor_runs", type=int, default=30)
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--qap_runs", type=int, default=30)
    parser.add_argument("--anchor_subsample", type=int, default=10_000)
    parser.add_argument("--initial_k", type=int, default=50)
    parser.add_argument("--refine_iters", type=int, default=100)
    parser.add_argument("--refine_sample", type=int, default=10_000)
    parser.add_argument("--refine_k", type=int, default=50)
    parser.add_argument("--refine2_clusters", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)

    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {embedding_dir}")
    embed_train = torch.load(embedding_dir / f"talk2dino_avg_self_attn_out_80000.pt", map_location="cpu", weights_only=True)
    embed_val = torch.load(embedding_dir / f"val_paired_8192.pt", map_location="cpu", weights_only=True)

    # train
    embed_A = embed_train["vision_feats"].float()
    embed_B = embed_train["text_feats"].float()
    embed_A, embed_B = pad_to_same_dim_v2(embed_A, embed_B)
    n = len(embed_A) - 8192

    mean_A = embed_A[:n].mean(dim=0)
    E_A1 = normalize(embed_A[: n // 2] - mean_A)
    E_A2 = normalize(embed_A[n // 2 : n] - mean_A)
    E_A3 = normalize(embed_A[n:] - mean_A)

    mean_B = embed_B[:n].mean(dim=0)
    E_B1 = normalize(embed_B[: n // 2] - mean_B)
    E_B2 = normalize(embed_B[n // 2 : n] - mean_B)
    E_B3 = normalize(embed_B[n:] - mean_B)

    #val
    embed_A_val = embed_val["vision_feats"].float()
    embed_B_val = embed_val["text_feats"].float()
    embed_A_val, embed_B_val = pad_to_same_dim_v2(embed_A_val, embed_B_val)
    mean_A_val = embed_A_val.mean(dim=0)
    E_A_val = normalize(embed_A_val - mean_A_val)
    mean_B_val = embed_B_val.mean(dim=0)
    E_B_val = normalize(embed_B_val - mean_B_val)

    X_train, Y_train = E_A1, E_B2
    X_eval, Y_eval = E_A_val, E_B_val

    print(f"[data] X_train={tuple(X_train.shape)}, Y_train={tuple(Y_train.shape)}")
    print(f"[data] X_eval={tuple(X_eval.shape)}, Y_eval={tuple(Y_eval.shape)}")

    print("[stage 1] approximate matching by aligned centroids")
    all_centers1, all_centers2 = [], []
    for i in trange(args.num_anchor_runs):
        centers1, centers2 = aligned_centroids(
            X_train,
            Y_train,
            n_qap_runs=args.qap_runs,
            n_clusters=args.num_clusters,
            subsample=args.anchor_subsample,
            seed=args.seed + i,
        )
        all_centers1.append(centers1)
        all_centers2.append(centers2)

    all_centers1 = torch.cat(all_centers1, dim=0)
    all_centers2 = torch.cat(all_centers2, dim=0)

    print("[stage 2] initial transformation")
    sim1 = cos_sim_matrix(X_train, all_centers1)
    sim2 = cos_sim_matrix(Y_train, all_centers2)
    sim_similarity = cos_sim_matrix(sim1, sim2)

    top_similar = sim_similarity.topk(dim=-1, k=args.initial_k).indices
    coefs = torch.ones(args.initial_k) / args.initial_k
    Y_matched = Y_train[top_similar].transpose(-1, -2) @ coefs

    W = train_orthogonal_linear(X_train, Y_matched)
    initial_score = eval_cos_score(X_eval, Y_eval, W)
    print(f"[initial] eval cosine = {initial_score:.4f}")

    print("[stage 3.1] refine-1 ICP-style nearest-neighbor refinement")
    history = {
        "source": args.source,
        "target": args.target,
        "seed": args.seed,
        "initial_cosine": initial_score,
        "refine1_cosine": [],
    }

    for it in trange(args.refine_iters):
        sample_points = X_train[torch.randperm(len(X_train))[:args.refine_sample]]
        sample_similarities = cos_sim_matrix(sample_points @ W, Y_train)
        neighbors = sample_similarities.topk(dim=-1, k=args.refine_k).indices
        sample_matched = Y_train[neighbors].mean(dim=1)

        W_new = train_orthogonal_linear(sample_points, sample_matched)
        W = (1 - args.alpha) * W + args.alpha * W_new

        score = eval_cos_score(X_eval, Y_eval, W)
        history["refine1_cosine"].append(score)
        print(f"[refine1] iter={it + 1:03d}, eval cosine={score:.4f}")

    print("[stage 3.2] refine-2 cluster-based correction")
    kmeans1 = KMeans(n_clusters=args.refine2_clusters, n_init=10, random_state=args.seed)
    kmeans1.fit(X_train)
    centers1 = tensor(kmeans1.cluster_centers_)

    kmeans2 = KMeans(
        n_clusters=args.refine2_clusters,
        init=(centers1 @ W).numpy(),
        n_init=1,
        random_state=args.seed,
    )
    kmeans2.fit(Y_train)
    centers2 = tensor(kmeans2.cluster_centers_)

    W_new = train_orthogonal_linear(centers1, centers2)
    W = (1 - args.alpha) * W + args.alpha * W_new

    refine2_score = eval_cos_score(X_eval, Y_eval, W)
    print(f"[refine2] eval cosine = {refine2_score:.4f}")

    print("[final] computing Top-1 Accuracy and Average Rank")
    top1, avg_rank = compute_final_metrics(X_eval, Y_eval, W, device="cuda")

    print(f"[result] Top-1 Accuracy = {top1:.6f}")
    print(f"[result] Average Rank   = {avg_rank:.6f}")

    history["refine2_cosine"] = refine2_score
    history["top1"] = top1
    history["avg_rank"] = avg_rank

    name = f"{args.source}_to_{args.target}_seed{args.seed}"
    torch.save(W, out_dir / f"W_{name}.pt")
    with open(out_dir / f"result_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[save] {out_dir / f'W_{name}.pt'}")
    print(f"[save] {out_dir / f'result_{name}.json'}")


if __name__ == "__main__":
    main()