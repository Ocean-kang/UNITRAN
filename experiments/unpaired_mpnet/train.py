"""Unpaired UNITRAN mini_vec2vec with independent banks and exact blockwise top-k.

Reuses src.training.unitran_mapping's aligned_centroids (KMeans+QAP),
cos_sim_matrix, orthogonal Procrustes, and eval_cos_score. The stage order,
optimizer parameters, and W updates are identical to train_mini_vec2vec.
Only the previous in-function X/Y half-splitting is removed: the two sides
were split by COCO image_id before extraction.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm.auto import trange

from common import assert_disjoint_and_paired, data_dir, dump_json, load_bank, load_config, root_path
from metrics import mean_selected, topk_cosine_indices
from src.training.unitran_mapping import (
    aligned_centroids, cos_sim_matrix, eval_cos_score,
    set_seed, tensor, train_orthogonal_linear,
)


def centered_unit(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return F.normalize(x - x.mean(dim=0, keepdim=True), dim=-1)


def fit(X: torch.Tensor, Y: torch.Tensor, Xe: torch.Tensor, Ye: torch.Tensor, cfg: dict):
    s = cfg["mapping"]
    set_seed(int(s["seed"]))
    n_clusters = int(s["num_clusters"])
    initial_k, refine_k = int(s["initial_k"]), int(s["refine_k"])
    refine2_clusters = int(s["refine2_clusters"])
    if min(len(X), len(Y)) < max(n_clusters, refine2_clusters, initial_k, refine_k):
        raise ValueError("Too few training images per side for configured cluster/topk counts. "
                         "Use more images or reduce the mapping params for a smoke test.")
    batch = int(s["similarity_batch_size"])
    device = str(s["similarity_device"])
    alpha = float(s["alpha"])
    X, Y, Xe, Ye = (centered_unit(z) for z in (X, Y, Xe, Ye))

    print("[stage 1] approximate matching by aligned centroids (original UNITRAN)")
    c1, c2 = [], []
    for i in trange(int(s["num_anchor_runs"])):
        a, b = aligned_centroids(X, Y,
            n_qap_runs=int(s["qap_runs"]), n_clusters=n_clusters,
            subsample=int(s["anchor_subsample"]), seed=int(s["seed"]) + i)
        c1.append(a)
        c2.append(b)
    c1, c2 = torch.cat(c1, dim=0), torch.cat(c2, dim=0)

    print("[stage 2] initial transformation via orthogonal Procrustes")
    sim1 = cos_sim_matrix(X, c1)
    sim2 = cos_sim_matrix(Y, c2)
    idx = topk_cosine_indices(sim1, sim2, initial_k, batch, device=device)
    matched = mean_selected(Y, idx, batch_size=batch)
    W = train_orthogonal_linear(X, matched)
    first_score = eval_cos_score(Xe, Ye, W)
    print(f"[initial] paired validation cosine = {first_score:.5f}")
    history = {
        "method": "mini_vec2vec", "source": "vision", "target": "text",
        "seed": s["seed"], "train_vision_rows": len(X), "train_text_rows": len(Y),
        "val_pairs": len(Xe), "initial_cosine": first_score, "refine1_cosine": [],
    }
    del sim1, sim2, idx, matched

    print("[stage 3.1] iterative nearest-neighbor refinement")
    for it in trange(int(s["refine_iters"])):
        sampled = X[torch.randperm(len(X))[:int(s["refine_sample"])]]
        nearest = topk_cosine_indices(sampled @ W, Y, refine_k, batch, device=device)
        y_match = mean_selected(Y, nearest, batch_size=batch)
        new_W = train_orthogonal_linear(sampled, y_match)
        W = (1 - alpha) * W + alpha * new_W
        score = eval_cos_score(Xe, Ye, W)
        history["refine1_cosine"].append(score)
        print(f"[refine1] iter={it+1:03d}, eval cosine={score:.5f}")

    print("[stage 3.2] cluster correction (original UNITRAN)")
    km1 = KMeans(n_clusters=refine2_clusters, n_init=10, random_state=int(s["seed"]))
    km1.fit(X)
    centers1 = tensor(km1.cluster_centers_)
    km2 = KMeans(n_clusters=refine2_clusters, init=(centers1 @ W).numpy(),
                 n_init=1, random_state=int(s["seed"]))
    km2.fit(Y)
    centers2 = tensor(km2.cluster_centers_)
    new_W = train_orthogonal_linear(centers1, centers2)
    W = (1 - alpha) * W + alpha * new_W
    history["refine2_cosine"] = eval_cos_score(Xe, Ye, W)
    return W, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/unpaired_mpnet/config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert_disjoint_and_paired(cfg)
    X = load_bank(cfg, "vision_train")["features"]
    Y = load_bank(cfg, "text_train")["features"]
    Xe = load_bank(cfg, "vision_val")["features"]
    Ye = load_bank(cfg, "text_val")["features"]
    out = root_path(cfg["paths"]["result_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["mapping"]["seed"])
    fname = f"mini_vec2vec_vision_to_text_seed{seed}"
    weight, result = out / f"W_{fname}.pt", out / f"result_{fname}.json"
    if (weight.exists() or result.exists()) and not args.force:
        raise FileExistsError(f"Output already exists in {out}; pass --force to rerun")
    W, history = fit(X, Y, Xe, Ye, cfg)
    torch.save(W.cpu(), weight)
    history["train_data_dir"] = str(data_dir(cfg))
    history["mapping_config"] = cfg["mapping"]
    dump_json(result, history)
    print(f"[save] {weight}")
    print(f"[save] {result}")


if __name__ == "__main__":
    main()
