import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
    if Path(path).suffix == ".npy":
        return np.load(path)
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


def sample_same_count(vision, text, max_points, seed):
    rng = torch.Generator().manual_seed(seed)
    n = min(len(vision), len(text), max_points)
    idx_v = torch.randperm(len(vision), generator=rng)[:n]
    idx_t = torch.randperm(len(text), generator=rng)[:n]
    return vision[idx_v], text[idx_t]


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


def project_2d(vision, text, method, seed):
    data = torch.cat([vision, text], dim=0).numpy()

    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        xy = reducer.fit_transform(data)
        extra = {"pca_explained_variance_ratio": reducer.explained_variance_ratio_.tolist()}
    elif method == "tsne":
        perplexity = min(30, max(5, (len(data) - 1) // 3))
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed)
        xy = reducer.fit_transform(data)
        extra = {"tsne_perplexity": perplexity}
    else:
        raise ValueError(f"Unknown method: {method}")

    return xy[: len(vision)], xy[len(vision) :], extra


def maybe_apply_w(vision, w_path):
    if w_path is None:
        return None
    W = safe_torch_load(Path(w_path)).float().cpu()
    if W.ndim != 2:
        raise ValueError(f"W must have shape (D, D), but got {tuple(W.shape)}")
    if vision.size(1) != W.size(0):
        raise ValueError(f"vision dim {vision.size(1)} does not match W input dim {W.size(0)}")
    return vision @ W


def draw_panel(ax, vision_xy, text_xy, title, point_size):
    ax.scatter(vision_xy[:, 0], vision_xy[:, 1], s=point_size, alpha=0.65, label="vision")
    ax.scatter(text_xy[:, 0], text_xy[:, 1], s=point_size, alpha=0.65, label="text")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)


def save_figure(before, after, args, meta):
    import matplotlib.pyplot as plt

    if after is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=args.dpi)
        draw_panel(ax, before[0], before[1], args.title, args.point_size)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=args.dpi)
        draw_panel(axes[0], before[0], before[1], "Original embeddings", args.point_size)
        draw_panel(axes[1], after[0], after[1], "After UNITRAN W", args.point_size)

    fig.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{args.prefix}_{args.method}_{args.preprocess}.png"
    meta_path = out_dir / f"{args.prefix}_{args.method}_{args.preprocess}.json"
    fig.savefig(fig_path, bbox_inches="tight")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {fig_path}")
    print(f"[save] {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", type=str, default=None)
    parser.add_argument("--vision_pt", type=str, default=None)
    parser.add_argument("--text_pt", type=str, default=None)
    parser.add_argument("--vision_key", type=str, default="vision_feats")
    parser.add_argument("--text_key", type=str, default="text_feats")
    parser.add_argument("--W", type=str, default=None)
    parser.add_argument("--method", type=str, choices=["pca", "tsne"], default="pca")
    parser.add_argument("--preprocess", type=str, choices=["raw", "l2", "unitran"], default="l2")
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--point_size", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/vis")
    parser.add_argument("--prefix", type=str, default="embeddings_2d")
    parser.add_argument("--title", type=str, default="Text/Vision embeddings")
    args = parser.parse_args()

    set_seed(args.seed)

    vision, text = load_pair(args)
    vision, text = sample_same_count(vision, text, args.max_points, args.seed)
    vision, text = preprocess_pair(vision, text, args.preprocess)

    print(f"[data] vision={tuple(vision.shape)}, text={tuple(text.shape)}")
    vision_xy, text_xy, before_meta = project_2d(vision, text, args.method, args.seed)

    after_vision = maybe_apply_w(vision, args.W)
    after = None
    after_meta = None
    if after_vision is not None:
        after_vision_xy, after_text_xy, after_meta = project_2d(after_vision, text, args.method, args.seed)
        after = (after_vision_xy, after_text_xy)

    meta = {
        "input_pt": args.input_pt,
        "vision_pt": args.vision_pt,
        "text_pt": args.text_pt,
        "vision_key": args.vision_key,
        "text_key": args.text_key,
        "W": args.W,
        "method": args.method,
        "preprocess": args.preprocess,
        "num_points_per_modality": len(vision),
        "before": before_meta,
        "after": after_meta,
    }
    save_figure((vision_xy, text_xy), after, args, meta)


if __name__ == "__main__":
    main()
