import argparse
import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange

from src.training.mini_vec2vec import (
    aligned_centroids, cos_sim_matrix, eval_cos_score, fit_unpaired,
    preprocess, set_seed, sim, tensor, train_orthogonal_linear,
)


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


def compute_final_metrics(X_eval, Y_eval, W=None, model=None, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if W is not None:
        X = X_eval.to(device) @ W.to(device)
    elif model is not None:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X_eval), 1024):
                outs.append(model.translate_source_to_target(X_eval[i:i + 1024].to(device)).cpu())
        X = torch.cat(outs, dim=0).to(device)
    else:
        X = X_eval.to(device)
    Y = Y_eval.to(device)

    sim_mat = cos_sim_matrix(X, Y, device=device)
    ranks = torch.argsort(torch.argsort(sim_mat, dim=-1), dim=-1).diagonal()

    top1 = (len(X_eval) - 1 == ranks).float().mean().item()
    avg_rank = (len(X_eval) - ranks.float().mean()).item()
    return top1, avg_rank


def load_feature_pair(path: Path, source_key: str, target_key: str):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if source_key not in obj or target_key not in obj:
        raise KeyError(f"{path} must contain '{source_key}' and '{target_key}'")
    return obj[source_key].float(), obj[target_key].float()


def first_existing_path(paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def resolve_data_paths(args):
    embedding_dir = Path(args.embedding_dir)

    if args.train_path:
        train_path = Path(args.train_path)
        if not train_path.exists():
            raise FileNotFoundError(f"Configured train_path does not exist: {train_path}")
    else:
        train_path = first_existing_path([
            embedding_dir / "local_density_eval_expanded.pt",
            embedding_dir / "local_density_eval_original.pt",
            embedding_dir / "talk2dino_avg_self_attn_out_80000.pt",
        ])

    if args.eval_path:
        eval_path = Path(args.eval_path)
        if not eval_path.exists():
            raise FileNotFoundError(f"Configured eval_path does not exist: {eval_path}")
    else:
        eval_path = first_existing_path([
            embedding_dir / "val_paired_8192.pt",
        ])

    if train_path is None:
        raise FileNotFoundError(f"Cannot find train embeddings under {embedding_dir}")
    return train_path, eval_path


def load_mapping_data(args, pad_dims: bool, normalize_data: bool = True):
    train_path, eval_path = resolve_data_paths(args)
    embed_A, embed_B = load_feature_pair(train_path, args.source_key, args.target_key)
    if pad_dims:
        embed_A, embed_B = pad_to_same_dim_v2(embed_A, embed_B)

    if eval_path is not None:
        embed_A_val, embed_B_val = load_feature_pair(eval_path, args.source_key, args.target_key)
        if pad_dims:
            embed_A_val, embed_B_val = pad_to_same_dim_v2(embed_A_val, embed_B_val)
        train_n = len(embed_A)
        if args.train_holdout > 0 and len(embed_A) > args.train_holdout:
            train_n = len(embed_A) - args.train_holdout
        train_A, train_B = embed_A[:train_n], embed_B[:train_n]
        eval_A, eval_B = embed_A_val, embed_B_val
    else:
        eval_size = min(args.train_holdout, max(1, len(embed_A) // 5)) if args.train_holdout > 0 else 0
        if eval_size > 0 and len(embed_A) > eval_size:
            train_A, train_B = embed_A[:-eval_size], embed_B[:-eval_size]
            eval_A, eval_B = embed_A[-eval_size:], embed_B[-eval_size:]
        else:
            train_A, train_B = embed_A, embed_B
            eval_A, eval_B = embed_A, embed_B

    if normalize_data:
        mean_A, mean_B = train_A.mean(dim=0), train_B.mean(dim=0)
        X_train, Y_train = normalize(train_A - mean_A), normalize(train_B - mean_B)
        X_eval, Y_eval = normalize(eval_A - mean_A), normalize(eval_B - mean_B)
    else:
        X_train, Y_train, X_eval, Y_eval = train_A, train_B, eval_A, eval_B

    print(f"[load] train={train_path}")
    if eval_path is not None:
        print(f"[load] eval={eval_path}")
    print(f"[data] X_train={tuple(X_train.shape)}, Y_train={tuple(Y_train.shape)}")
    print(f"[data] X_eval={tuple(X_eval.shape)}, Y_eval={tuple(Y_eval.shape)}")
    return X_train, Y_train, X_eval, Y_eval, {"train_path": str(train_path), "eval_path": str(eval_path) if eval_path else None}


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int, residual: bool = True):
        super().__init__()
        layers = []
        cur = in_dim
        for _ in range(max(1, depth - 1)):
            layers += [nn.Linear(cur, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)]
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)
        self.residual = residual and in_dim == out_dim

    def forward(self, x):
        y = self.net(x)
        return x + y if self.residual else y


class Vec2VecTranslator(nn.Module):
    def __init__(self, source_dim: int, target_dim: int, latent_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.source_in = ResidualMLP(source_dim, latent_dim, hidden_dim, depth, residual=False)
        self.target_in = ResidualMLP(target_dim, latent_dim, hidden_dim, depth, residual=False)
        self.backbone = ResidualMLP(latent_dim, latent_dim, hidden_dim, depth, residual=True)
        self.source_out = ResidualMLP(latent_dim, source_dim, hidden_dim, depth, residual=False)
        self.target_out = ResidualMLP(latent_dim, target_dim, hidden_dim, depth, residual=False)

    def source_latent(self, x):
        return self.backbone(self.source_in(x))

    def target_latent(self, y):
        return self.backbone(self.target_in(y))

    def reconstruct_source(self, x):
        return normalize(self.source_out(self.source_latent(x)))

    def reconstruct_target(self, y):
        return normalize(self.target_out(self.target_latent(y)))

    def translate_source_to_target(self, x):
        return normalize(self.target_out(self.source_latent(x)))

    def translate_target_to_source(self, y):
        return normalize(self.source_out(self.target_latent(y)))


class Discriminator(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers = []
        cur = dim
        for _ in range(max(1, depth - 1)):
            layers += [nn.Linear(cur, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)]
            cur = hidden_dim
        layers.append(nn.Linear(cur, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def sample_batch(X, batch_size, device):
    idx = torch.randint(0, len(X), (batch_size,))
    return X[idx].to(device)


def cosine_loss(x, y):
    return 1 - F.cosine_similarity(x, y, dim=-1).mean()


def vsp_loss(real, fake):
    real = normalize(real)
    fake = normalize(fake)
    return (real @ real.T - fake @ fake.T).abs().mean()


def disc_loss_fn(disc, real, fake):
    real_logits = disc(real.detach())
    fake_logits = disc(fake.detach())
    return (
        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) +
        F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    ) / 2


def gen_loss_fn(disc, fake):
    logits = disc(fake)
    return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))


def train_mini_vec2vec(args):
    """Compatibility entry for legacy row-based banks; core fitting is shared."""
    warnings.warn("Legacy feature-row protocol cannot establish image-ID disjointness. "
                  "Use experiments/unpaired.py for COCO2014 experiments.", RuntimeWarning)
    X_all, Y_all, X_eval, Y_eval, meta = load_mapping_data(args, pad_dims=True, normalize_data=False)
    n = min(len(X_all), len(Y_all))
    X_train, Y_train = X_all[:n // 2], Y_all[n // 2:]
    keys = ("seed", "num_anchor_runs", "num_clusters", "qap_runs", "anchor_subsample",
            "initial_k", "refine_iters", "refine_sample", "refine_k", "refine2_clusters", "alpha")
    params = {key: getattr(args, key) for key in keys}
    params.update(similarity_batch_size=128, similarity_device="cpu")
    state, history = fit_unpaired(X_train, Y_train, params)
    xe = preprocess(X_eval, state["mean_source"], state["dim"])
    ye = preprocess(Y_eval, state["mean_target"], state["dim"])
    weight = state["weight"]
    top1, avg_rank = compute_final_metrics(xe, ye, W=weight, device=args.device)
    history.update(method=args.method, source=args.source, target=args.target, seed=args.seed,
                   refine2_cosine=eval_cos_score(xe, ye, weight), top1=top1, avg_rank=avg_rank, **meta)
    name = f"{args.method}_{args.source}_to_{args.target}_seed{args.seed}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(weight, out / f"W_{name}.pt")
    torch.save(state, out / f"preprocessing_{name}.pt")
    (out / f"result_{name}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def train_vec2vec(args):
    X_train, Y_train, X_eval, Y_eval, meta = load_mapping_data(args, pad_dims=False)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = Vec2VecTranslator(
        source_dim=X_train.size(1),
        target_dim=Y_train.size(1),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)
    d_source = Discriminator(X_train.size(1), args.discriminator_dim, args.discriminator_depth).to(device)
    d_target = Discriminator(Y_train.size(1), args.discriminator_dim, args.discriminator_depth).to(device)
    d_latent = Discriminator(args.latent_dim, args.discriminator_dim, args.discriminator_depth).to(device)

    opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(
        list(d_source.parameters()) + list(d_target.parameters()) + list(d_latent.parameters()),
        lr=args.discriminator_lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "method": args.method,
        "source": args.source,
        "target": args.target,
        "seed": args.seed,
        **meta,
        "cosine": [],
        "loss": [],
    }
    steps_per_epoch = args.steps_per_epoch or max(1, min(len(X_train), len(Y_train)) // args.batch_size)

    print("[stage] adversarial vec2vec training")
    for epoch in trange(args.epochs):
        loss_total = 0.0
        for _ in range(steps_per_epoch):
            x = sample_batch(X_train, args.batch_size, device)
            y = sample_batch(Y_train, args.batch_size, device)

            x2y = model.translate_source_to_target(x)
            y2x = model.translate_target_to_source(y)
            x_lat = model.source_latent(x)
            y_lat = model.target_latent(y)

            opt_d.zero_grad()
            d_loss = (
                disc_loss_fn(d_source, x, y2x) +
                disc_loss_fn(d_target, y, x2y) +
                disc_loss_fn(d_latent, x_lat, y_lat)
            )
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(d_source.parameters()) + list(d_target.parameters()) + list(d_latent.parameters()), args.max_grad_norm)
            opt_d.step()

            x_rec = model.reconstruct_source(x)
            y_rec = model.reconstruct_target(y)
            x2y = model.translate_source_to_target(x)
            y2x = model.translate_target_to_source(y)
            x_cycle = model.translate_target_to_source(x2y)
            y_cycle = model.translate_source_to_target(y2x)
            x_lat = model.source_latent(x)
            y_lat = model.target_latent(y)

            rec_loss = cosine_loss(x_rec, x) + cosine_loss(y_rec, y)
            cc_loss = cosine_loss(x_cycle, x) + cosine_loss(y_cycle, y)
            geom_loss = vsp_loss(y, x2y) + vsp_loss(x, y2x)
            adv_loss = gen_loss_fn(d_target, x2y) + gen_loss_fn(d_source, y2x) + gen_loss_fn(d_latent, y_lat)
            loss = (
                args.lambda_rec * rec_loss +
                args.lambda_cc * cc_loss +
                args.lambda_vsp * geom_loss +
                args.lambda_adv * adv_loss
            )

            opt_g.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt_g.step()
            loss_total += loss.item()

        with torch.no_grad():
            pred = model.translate_source_to_target(X_eval.to(device)).cpu()
            score = torch.cosine_similarity(pred, Y_eval, dim=-1).mean().item()
        history["cosine"].append(score)
        history["loss"].append(loss_total / steps_per_epoch)
        print(f"[vec2vec] epoch={epoch + 1:03d}, loss={history['loss'][-1]:.4f}, eval cosine={score:.4f}")

    print("[final] computing Top-1 Accuracy and Average Rank")
    top1, avg_rank = compute_final_metrics(X_eval, Y_eval, model=model, device=device)
    history["final_cosine"] = history["cosine"][-1] if history["cosine"] else None
    history["top1"] = top1
    history["avg_rank"] = avg_rank
    print(f"[result] Top-1 Accuracy = {top1:.6f}")
    print(f"[result] Average Rank   = {avg_rank:.6f}")

    name = f"{args.method}_{args.source}_to_{args.target}_seed{args.seed}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "source_dim": X_train.size(1),
        "target_dim": Y_train.size(1),
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "source_key": args.source_key,
        "target_key": args.target_key,
    }, out_dir / f"model_{name}.pt")
    with open(out_dir / f"result_{name}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[save] {out_dir / f'model_{name}.pt'}")
    print(f"[save] {out_dir / f'result_{name}.json'}")


METHODS = {
    "mini_vec2vec": train_mini_vec2vec,
    "vec2vec": train_vec2vec,
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), default="mini_vec2vec")
    parser.add_argument("--embedding_dir", type=str, required=True)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--source_key", type=str, default="vision_feats")
    parser.add_argument("--target_key", type=str, default="text_feats")
    parser.add_argument("--source", type=str, default="clip")
    parser.add_argument("--target", type=str, default="dinov2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_holdout", type=int, default=8192)
    parser.add_argument("--out_dir", type=str, default="outputs")

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

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--discriminator_dim", type=int, default=1024)
    parser.add_argument("--discriminator_depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discriminator_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_rec", type=float, default=5.0)
    parser.add_argument("--lambda_cc", type=float, default=5.0)
    parser.add_argument("--lambda_vsp", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    METHODS[args.method](args)


if __name__ == "__main__":
    main()
