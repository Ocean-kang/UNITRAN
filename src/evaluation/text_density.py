#!/usr/bin/env python3
"""
Build a small evaluation set for checking whether LLM-expanded text atoms better match
DINOv2 patch-token distribution than original COCO captions.

Stages:
  sample   -> sample image ids/file names/captions from COCO annotation and feature images.jsonl
  features -> encode original captions + expanded atoms with sentence-transformers, sample patch tokens
  report   -> compare patch-token distribution with original-text and expanded-text distributions

No cluster-guided text generation is used in this script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_ann(coco_root: Path, split: str) -> Path:
    for p in [coco_root / "annotations" / f"captions_{split}.json", coco_root / f"captions_{split}.json"]:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find captions_{split}.json under {coco_root}")


def load_coco_captions(ann_path: Path) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    ann = read_json(ann_path)
    file_by_id: Dict[int, str] = {}
    for img in ann.get("images", []):
        file_by_id[int(img["id"])] = str(img["file_name"])
    caps_by_id: Dict[int, List[str]] = {}
    for a in ann.get("annotations", []):
        caps_by_id.setdefault(int(a["image_id"]), []).append(str(a["caption"]).strip())
    return file_by_id, caps_by_id


def stage_sample(args: argparse.Namespace) -> None:
    feature_dir = Path(args.feature_dir)
    images_jsonl = feature_dir / "images.jsonl"
    if not images_jsonl.exists():
        raise FileNotFoundError(f"Missing {images_jsonl}. Run extract_coco_dinov2_patch.py first.")

    feature_rows = load_jsonl(images_jsonl)
    ann_path = Path(args.ann_path) if args.ann_path else find_ann(Path(args.coco_root), args.split)
    _, caps_by_id = load_coco_captions(ann_path)

    candidates = []
    for r in feature_rows:
        image_id = int(r["image_id"])
        caps = caps_by_id.get(image_id, [])
        if caps:
            candidates.append({
                "index": int(r["index"]),
                "image_id": image_id,
                "file_name": r["file_name"],
                "path": r.get("path", ""),
                "captions": caps[: args.max_captions],
            })

    if len(candidates) < args.num_images:
        raise ValueError(f"Only {len(candidates)} feature rows have captions, requested {args.num_images}")

    rng = random.Random(args.seed)
    if args.sample_mode == "first":
        sample = candidates[: args.num_images]
    else:
        sample = rng.sample(candidates, args.num_images)
        sample.sort(key=lambda x: x["index"])

    write_jsonl(Path(args.out_jsonl), sample)
    print(f"[sample] candidates={len(candidates)}, selected={len(sample)}")
    print(f"[save] {args.out_jsonl}")


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s+", text.strip())
    return [p.strip(" -\t\n\r") for p in parts if len(p.strip()) >= 3]


def load_recaption_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = load_jsonl(path)
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        out[int(r["image_id"])] = r
    return out


def collect_texts(sample_rows: List[Dict[str, Any]], recaption_rows: Dict[int, Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    orig_texts: List[str] = []
    orig_meta: List[Dict[str, Any]] = []
    exp_texts: List[str] = []
    exp_meta: List[Dict[str, Any]] = []

    for r in sample_rows:
        image_id = int(r["image_id"])
        for j, c in enumerate(r["captions"]):
            c = str(c).strip()
            if c:
                orig_texts.append(c)
                orig_meta.append({"image_id": image_id, "file_name": r["file_name"], "source": "orig_caption", "caption_id": j, "text": c})

        rr = recaption_rows.get(image_id)
        if rr is None:
            # Fallback: evaluate original-caption sentence distribution as expanded if recaption is missing.
            atoms = list(r["captions"])
            long_caption = " ".join(r["captions"])
        else:
            atoms = rr.get("atoms") or []
            long_caption = str(rr.get("long_caption", ""))
            if not atoms:
                atoms = split_sentences(long_caption)
        for j, a in enumerate(atoms):
            a = str(a).strip()
            if a:
                exp_texts.append(a)
                exp_meta.append({"image_id": image_id, "file_name": r["file_name"], "source": "expanded_atom", "atom_id": j, "text": a})

    return orig_texts, orig_meta, exp_texts, exp_meta


def encode_texts(texts: List[str], model_name: str, batch_size: int, device: str) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return torch.from_numpy(emb).float()


def resolve_mmap_path(feature_dir: Path, meta: Dict[str, Any]) -> Path:
    p = Path(meta["mmap_path"])
    if p.exists():
        return p
    # Older metadata may be written relative to repo root or feature dir.
    cand = feature_dir / "patch_tokens" / p.name
    if cand.exists():
        return cand
    cand = feature_dir / p
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Cannot resolve mmap path from metadata: {meta['mmap_path']}")


def load_shard_index(feature_dir: Path) -> Dict[int, Tuple[Path, int, Tuple[int, int, int], str]]:
    patch_dir = feature_dir / "patch_tokens"
    out: Dict[int, Tuple[Path, int, Tuple[int, int, int], str]] = {}
    for meta_path in sorted(patch_dir.glob("shard_*.meta.json")):
        meta = read_json(meta_path)
        mmap_path = resolve_mmap_path(feature_dir, meta)
        shape = tuple(int(x) for x in meta["shape"])
        dtype = str(meta.get("dtype", "float16"))
        for local_i, global_i in enumerate(meta["global_indices"]):
            out[int(global_i)] = (mmap_path, local_i, shape, dtype)
    if not out:
        raise FileNotFoundError(f"No shard_*.meta.json found in {patch_dir}")
    return out


def sample_patch_tokens(feature_dir: Path, sample_rows: List[Dict[str, Any]], patches_per_image: int, seed: int) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    shard_index = load_shard_index(feature_dir)
    rng = np.random.default_rng(seed)
    grouped: Dict[Path, List[Tuple[int, Dict[str, Any]]]] = {}
    shard_info: Dict[Path, Tuple[Tuple[int, int, int], str]] = {}
    for r in sample_rows:
        global_i = int(r["index"])
        if global_i not in shard_index:
            raise KeyError(f"global index {global_i} not found in patch shard metadata")
        mmap_path, local_i, shape, dtype = shard_index[global_i]
        grouped.setdefault(mmap_path, []).append((local_i, r))
        shard_info[mmap_path] = (shape, dtype)

    feats: List[torch.Tensor] = []
    meta_rows: List[Dict[str, Any]] = []
    for mmap_path, items in tqdm(grouped.items(), desc="sample patch tokens"):
        shape, dtype = shard_info[mmap_path]
        arr = np.memmap(mmap_path, dtype=np.float16 if dtype == "float16" else np.float32, mode="r", shape=shape)
        num_patches = shape[1]
        k = min(patches_per_image, num_patches)
        for local_i, r in items:
            patch_ids = rng.choice(num_patches, size=k, replace=False)
            patch = np.asarray(arr[local_i, patch_ids], dtype=np.float32)
            patch_t = F.normalize(torch.from_numpy(patch).float(), dim=1, eps=1e-12)
            feats.append(patch_t)
            for pid in patch_ids:
                meta_rows.append({
                    "image_id": int(r["image_id"]),
                    "file_name": r["file_name"],
                    "global_index": int(r["index"]),
                    "local_shard_index": int(local_i),
                    "patch_id": int(pid),
                })
    return torch.cat(feats, dim=0), meta_rows


def stage_features(args: argparse.Namespace) -> None:
    sample_rows = load_jsonl(Path(args.sample_jsonl))
    recaption_rows = load_recaption_rows(Path(args.recaption_jsonl)) if args.recaption_jsonl else {}
    orig_texts, orig_meta, exp_texts, exp_meta = collect_texts(sample_rows, recaption_rows)

    print(f"[text] original captions={len(orig_texts)}, expanded atoms={len(exp_texts)}")
    if not exp_texts:
        raise ValueError("No expanded atoms found. Check --recaption_jsonl.")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    orig_feats = encode_texts(orig_texts, args.text_model, args.text_batch_size, device)
    exp_feats = encode_texts(exp_texts, args.text_model, args.text_batch_size, device)
    vision_feats, vision_meta = sample_patch_tokens(Path(args.feature_dir), sample_rows, args.patches_per_image, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"vision_feats": vision_feats, "text_feats": orig_feats}, out_dir / "density_eval_original.pt")
    torch.save({"vision_feats": vision_feats, "text_feats": exp_feats}, out_dir / "density_eval_expanded.pt")
    torch.save({"vision_feats": vision_feats}, out_dir / "vision_patch_sample.pt")
    torch.save({"text_feats": orig_feats}, out_dir / "text_original.pt")
    torch.save({"text_feats": exp_feats}, out_dir / "text_expanded.pt")
    write_jsonl(out_dir / "text_original_meta.jsonl", orig_meta)
    write_jsonl(out_dir / "text_expanded_meta.jsonl", exp_meta)
    write_jsonl(out_dir / "vision_patch_meta.jsonl", vision_meta)
    write_json(out_dir / "feature_config.json", {
        "sample_jsonl": args.sample_jsonl,
        "recaption_jsonl": args.recaption_jsonl,
        "feature_dir": args.feature_dir,
        "text_model": args.text_model,
        "num_images": len(sample_rows),
        "num_original_texts": len(orig_texts),
        "num_expanded_texts": len(exp_texts),
        "num_vision_patches": int(vision_feats.shape[0]),
        "vision_dim": int(vision_feats.shape[1]),
        "text_dim": int(exp_feats.shape[1]),
        "patches_per_image": args.patches_per_image,
    })
    print(f"[save] {out_dir}")


def safe_load_tensor(path: Path, key: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    x = obj[key] if isinstance(obj, dict) else obj
    return x.float().cpu()


def maybe_equal_dim(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.shape[1] == b.shape[1]:
        return a, b
    d = max(a.shape[1], b.shape[1])
    def pad(x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == d:
            return x
        y = torch.zeros(x.shape[0], d, dtype=x.dtype)
        y[:, : x.shape[1]] = x
        return y
    return pad(a), pad(b)


def sample_rows_tensor(x: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    if n <= 0 or len(x) <= n:
        return x
    g = torch.Generator().manual_seed(seed)
    return x[torch.randperm(len(x), generator=g)[:n]]


def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1, eps=1e-12)
    y = F.normalize(y, dim=1, eps=1e-12)
    return (1 - x @ y.T).clamp_min(0)


def pairwise_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x * x).sum(1, keepdim=True) + (y * y).sum(1, keepdim=True).T - 2 * x @ y.T).clamp_min(0)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    xx = pairwise_sq(x, x)
    yy = pairwise_sq(y, y)
    xy = pairwise_sq(x, y)
    vals = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])
    vals = vals[vals > 0]
    sigma = torch.sqrt(torch.median(vals) / 2).item() if vals.numel() else 1.0
    sigma = max(float(sigma), 1e-6)
    gamma = 1.0 / (2 * sigma * sigma)
    val = torch.exp(-gamma * xx).mean() + torch.exp(-gamma * yy).mean() - 2 * torch.exp(-gamma * xy).mean()
    return {"mmd_rbf": float(val.item()), "mmd_sigma": sigma}


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_proj: int, seed: int) -> float:
    x, y = maybe_equal_dim(x, y)
    g = torch.Generator().manual_seed(seed)
    d = x.shape[1]
    dirs = torch.randn(n_proj, d, generator=g)
    dirs = F.normalize(dirs, dim=1, eps=1e-12)
    vals = []
    n = min(len(x), len(y))
    for v in dirs:
        xp = torch.sort(x @ v).values
        yp = torch.sort(y @ v).values
        if len(xp) != n:
            xp = xp[torch.linspace(0, len(xp) - 1, n).long()]
        if len(yp) != n:
            yp = yp[torch.linspace(0, len(yp) - 1, n).long()]
        vals.append(torch.mean(torch.abs(xp - yp)))
    return float(torch.stack(vals).mean().item())


def knn_radii(x: torch.Tensor, k: int) -> torch.Tensor:
    dist = cosine_dist(x, x)
    dist.fill_diagonal_(float("inf"))
    k = min(k, len(x) - 1)
    return dist.kthvalue(k, dim=1).values


def precision_recall_coverage(vision: torch.Tensor, text: torch.Tensor, k: int) -> Dict[str, float]:
    v = F.normalize(vision, dim=1, eps=1e-12)
    t = F.normalize(text, dim=1, eps=1e-12)
    rv = knn_radii(v, k)
    rt = knn_radii(t, k)
    d = cosine_dist(v, t)
    t_to_v_dist, t_to_v = d.min(dim=0)
    v_to_t_dist, v_to_t = d.min(dim=1)
    precision = (t_to_v_dist <= rv[t_to_v]).float().mean().item()
    recall = (v_to_t_dist <= rt[v_to_t]).float().mean().item()
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision_text_in_vision": float(precision),
        "recall_vision_in_text": float(recall),
        "precision_recall_f1": float(f1),
        "coverage_vision_by_text": float(t_to_v.unique().numel() / len(v)),
        "coverage_text_by_vision": float(v_to_t.unique().numel() / len(t)),
    }


def pca_participation_ratio(x: torch.Tensor) -> float:
    x = x - x.mean(0, keepdim=True)
    # Use covariance eigenvalues; sample size is capped in report stage.
    s = torch.linalg.svdvals(x)
    eig = s * s
    return float((eig.sum() ** 2 / (eig.square().sum().clamp_min(1e-12))).item())


def metrics(vision: torch.Tensor, text: torch.Tensor, args: argparse.Namespace, seed_offset: int = 0) -> Dict[str, Any]:
    vision, text = maybe_equal_dim(vision, text)
    vision = sample_rows_tensor(vision, args.max_points, args.seed + seed_offset)
    text = sample_rows_tensor(text, args.max_points, args.seed + 100 + seed_offset)
    vision = F.normalize(vision, dim=1, eps=1e-12)
    text = F.normalize(text, dim=1, eps=1e-12)
    out: Dict[str, Any] = {
        "num_vision": int(len(vision)),
        "num_text": int(len(text)),
        "vision_pr": pca_participation_ratio(vision),
        "text_pr": pca_participation_ratio(text),
    }
    out.update(mmd_rbf(vision, text))
    out["sliced_wasserstein"] = sliced_wasserstein(vision, text, args.sw_proj, args.seed + seed_offset)
    out.update(precision_recall_coverage(vision, text, args.knn_k))
    out["participation_ratio_gap_abs"] = abs(out["vision_pr"] - out["text_pr"])
    return out


def percent_change(new: float, old: float, lower_is_better: bool) -> float:
    if abs(old) < 1e-12:
        return 0.0
    if lower_is_better:
        return 100.0 * (old - new) / abs(old)
    return 100.0 * (new - old) / abs(old)


def stage_report(args: argparse.Namespace) -> None:
    feature_dir = Path(args.feature_dir) if getattr(args, "feature_dir", None) else Path(args.out_dir)
    out_dir = Path(args.out_dir)
    vision = safe_load_tensor(feature_dir / "vision_patch_sample.pt", "vision_feats")
    orig = safe_load_tensor(feature_dir / "text_original.pt", "text_feats")
    exp = safe_load_tensor(feature_dir / "text_expanded.pt", "text_feats")
    m_orig = metrics(vision, orig, args, 0)
    m_exp = metrics(vision, exp, args, 1)
    summary = {
        "feature_dir": str(feature_dir),
        "original": m_orig,
        "expanded": m_exp,
        "improvement_percent": {
            "mmd_rbf": percent_change(m_exp["mmd_rbf"], m_orig["mmd_rbf"], True),
            "sliced_wasserstein": percent_change(m_exp["sliced_wasserstein"], m_orig["sliced_wasserstein"], True),
            "precision_recall_f1": percent_change(m_exp["precision_recall_f1"], m_orig["precision_recall_f1"], False),
            "coverage_vision_by_text": percent_change(m_exp["coverage_vision_by_text"], m_orig["coverage_vision_by_text"], False),
            "participation_ratio_gap_abs": percent_change(m_exp["participation_ratio_gap_abs"], m_orig["participation_ratio_gap_abs"], True),
        },
        "interpretation": {
            "better_when_lower": ["mmd_rbf", "sliced_wasserstein", "participation_ratio_gap_abs"],
            "better_when_higher": ["precision_recall_f1", "coverage_vision_by_text", "recall_vision_in_text"],
        },
    }
    write_json(out_dir / "density_report.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[load] {feature_dir}")
    print(f"[save] {out_dir / 'density_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO text/patch distribution density evaluation")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("sample")
    p.add_argument("--coco_root", default="data/mscoco14", type=str)
    p.add_argument("--split", default="train2014", type=str)
    p.add_argument("--ann_path", default=None, type=str)
    p.add_argument("--feature_dir", default="feature/coco2014_dinov2_vitb14_reg_448", type=str)
    p.add_argument("--num_images", default=1000, type=int)
    p.add_argument("--max_captions", default=5, type=int)
    p.add_argument("--sample_mode", choices=["first", "random"], default="random")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--out_jsonl", default="outputs/text_density_eval/sample_1000.jsonl", type=str)

    p = sub.add_parser("features")
    p.add_argument("--sample_jsonl", default="outputs/text_density_eval/sample_1000.jsonl", type=str)
    p.add_argument("--recaption_jsonl", default=None, type=str)
    p.add_argument("--feature_dir", default="feature/coco2014_dinov2_vitb14_reg_448", type=str)
    p.add_argument("--text_model", default="sentence-transformers/all-mpnet-base-v2", type=str)
    p.add_argument("--text_batch_size", default=128, type=int)
    p.add_argument("--patches_per_image", default=64, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--out_dir", default="outputs/text_density_eval", type=str)

    p = sub.add_parser("report")
    p.add_argument("--feature_dir", default="feature/density/coco2014_patch", type=str)
    p.add_argument("--out_dir", default="outputs/density/coco2014_patch", type=str)
    p.add_argument("--max_points", default=20000, type=int)
    p.add_argument("--knn_k", default=5, type=int)
    p.add_argument("--sw_proj", default=128, type=int)
    p.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    set_seed(getattr(args, "seed", 0))
    if args.stage == "sample":
        stage_sample(args)
    elif args.stage == "features":
        stage_features(args)
    elif args.stage == "report":
        stage_report(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
