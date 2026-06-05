#!/usr/bin/env python3
"""
Small-sample text/vision density evaluation using per-image local patch centers.

Flow:
  sample   -> sample image ids/file names/captions from a local-center feature dir
  features -> load all local centers and encode exactly K text atoms per image
  report   -> compare local-center distribution with original-caption and expanded-atom distributions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from unitran.evaluation.text_density import (
    encode_texts,
    find_ann,
    load_coco_captions,
    load_jsonl,
    load_recaption_rows,
    metrics,
    percent_change,
    read_json,
    split_sentences,
    write_json,
    write_jsonl,
    set_seed,
)


def stage_sample(args: argparse.Namespace) -> None:
    feature_dir = Path(args.feature_dir)
    images_jsonl = feature_dir / "images.jsonl"
    if not images_jsonl.exists():
        raise FileNotFoundError(f"Missing {images_jsonl}. Build local centers first.")

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
        raise ValueError(f"Only {len(candidates)} local-center rows have captions, requested {args.num_images}")

    if args.sample_mode == "first":
        sample = candidates[: args.num_images]
    else:
        import random
        rng = random.Random(args.seed)
        sample = rng.sample(candidates, args.num_images)
        sample.sort(key=lambda x: x["index"])

    write_jsonl(Path(args.out_jsonl), sample)
    print(f"[sample] candidates={len(candidates)}, selected={len(sample)}")
    print(f"[save] {args.out_jsonl}")


def resolve_mmap_path(feature_dir: Path, meta: Dict[str, Any]) -> Path:
    p = Path(meta["mmap_path"])
    if p.exists():
        return p
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


def infer_local_k(local_feature_dir: Path) -> int:
    shape_path = local_feature_dir / "patch_tokens_shape.json"
    if shape_path.exists():
        shape = read_json(shape_path)
        full_shape = shape.get("full_shape") or []
        if len(full_shape) >= 2:
            return int(full_shape[1])

    meta_paths = sorted((local_feature_dir / "patch_tokens").glob("shard_*.meta.json"))
    if not meta_paths:
        raise FileNotFoundError(f"Cannot infer local_k under {local_feature_dir / 'patch_tokens'}")
    meta = read_json(meta_paths[0])
    return int(meta["shape"][1])


def load_local_centers(
    local_feature_dir: Path,
    sample_rows: List[Dict[str, Any]],
    expected_k: int,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    shard_index = load_shard_index(local_feature_dir)
    grouped: Dict[Path, List[Tuple[int, Dict[str, Any]]]] = {}
    shard_info: Dict[Path, Tuple[Tuple[int, int, int], str]] = {}

    for r in sample_rows:
        global_i = int(r["index"])
        if global_i not in shard_index:
            raise KeyError(f"global index {global_i} not found in local-center shard metadata")
        mmap_path, local_i, shape, dtype = shard_index[global_i]
        if int(shape[1]) != expected_k:
            raise ValueError(f"Expected {expected_k} local centers per image, got {shape[1]} in {mmap_path}")
        grouped.setdefault(mmap_path, []).append((local_i, r))
        shard_info[mmap_path] = (shape, dtype)

    feats: List[torch.Tensor] = []
    meta_rows: List[Dict[str, Any]] = []
    for mmap_path, items in grouped.items():
        shape, dtype = shard_info[mmap_path]
        arr = np.memmap(mmap_path, dtype=np.float16 if dtype == "float16" else np.float32, mode="r", shape=shape)
        for local_i, r in items:
            centers = np.asarray(arr[local_i], dtype=np.float32)
            feats.append(torch.from_numpy(centers).float())
            for center_id in range(expected_k):
                meta_rows.append({
                    "image_id": int(r["image_id"]),
                    "file_name": r["file_name"],
                    "global_index": int(r["index"]),
                    "local_center_id": int(center_id),
                })
        del arr

    return torch.cat(feats, dim=0), meta_rows


def exact_count(items: List[str], k: int) -> List[str]:
    clean = [str(x).strip() for x in items if str(x).strip()]
    if not clean:
        raise ValueError("Cannot build exact-count text atoms from an empty list")
    if len(clean) >= k:
        return clean[:k]
    return [clean[i % len(clean)] for i in range(k)]


def collect_exact_texts(
    sample_rows: List[Dict[str, Any]],
    recaption_rows: Dict[int, Dict[str, Any]],
    atoms_per_image: int,
) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    orig_texts: List[str] = []
    orig_meta: List[Dict[str, Any]] = []
    exp_texts: List[str] = []
    exp_meta: List[Dict[str, Any]] = []

    for r in sample_rows:
        image_id = int(r["image_id"])
        orig_atoms = exact_count(list(r["captions"]), atoms_per_image)
        for j, text in enumerate(orig_atoms):
            orig_texts.append(text)
            orig_meta.append({"image_id": image_id, "file_name": r["file_name"], "source": "orig_caption_exact", "atom_id": j, "text": text})

        rr = recaption_rows.get(image_id)
        if rr is None:
            atom_candidates = list(r["captions"])
        else:
            atom_candidates = rr.get("atoms") or split_sentences(str(rr.get("long_caption", ""))) or list(r["captions"])
        exp_atoms = exact_count(list(atom_candidates), atoms_per_image)
        for j, text in enumerate(exp_atoms):
            exp_texts.append(text)
            exp_meta.append({"image_id": image_id, "file_name": r["file_name"], "source": "expanded_atom_exact", "atom_id": j, "text": text})

    return orig_texts, orig_meta, exp_texts, exp_meta


def stage_features(args: argparse.Namespace) -> None:
    local_feature_dir = Path(args.local_feature_dir)
    sample_rows = load_jsonl(Path(args.sample_jsonl))
    atoms_per_image = int(args.atoms_per_image) if args.atoms_per_image > 0 else infer_local_k(local_feature_dir)
    recaption_rows = load_recaption_rows(Path(args.recaption_jsonl)) if args.recaption_jsonl else {}

    vision_feats, vision_meta = load_local_centers(local_feature_dir, sample_rows, atoms_per_image)
    orig_texts, orig_meta, exp_texts, exp_meta = collect_exact_texts(sample_rows, recaption_rows, atoms_per_image)

    if len(vision_feats) != len(orig_texts) or len(vision_feats) != len(exp_texts):
        raise RuntimeError("Internal count mismatch between local centers and text atoms")

    print(f"[vision] local centers={tuple(vision_feats.shape)}")
    print(f"[text] original atoms={len(orig_texts)}, expanded atoms={len(exp_texts)}, atoms_per_image={atoms_per_image}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    orig_feats = encode_texts(orig_texts, args.text_model, args.text_batch_size, device)
    exp_feats = encode_texts(exp_texts, args.text_model, args.text_batch_size, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"vision_feats": vision_feats, "text_feats": orig_feats}, out_dir / "local_density_eval_original.pt")
    torch.save({"vision_feats": vision_feats, "text_feats": exp_feats}, out_dir / "local_density_eval_expanded.pt")
    torch.save({"vision_feats": vision_feats}, out_dir / "vision_local_centers.pt")
    torch.save({"text_feats": orig_feats}, out_dir / "text_original_exact.pt")
    torch.save({"text_feats": exp_feats}, out_dir / "text_expanded_exact.pt")
    write_jsonl(out_dir / "vision_local_center_meta.jsonl", vision_meta)
    write_jsonl(out_dir / "text_original_exact_meta.jsonl", orig_meta)
    write_jsonl(out_dir / "text_expanded_exact_meta.jsonl", exp_meta)
    write_json(out_dir / "feature_config.json", {
        "sample_jsonl": args.sample_jsonl,
        "recaption_jsonl": args.recaption_jsonl,
        "local_feature_dir": args.local_feature_dir,
        "text_model": args.text_model,
        "num_images": len(sample_rows),
        "atoms_per_image": atoms_per_image,
        "num_vision_centers": int(vision_feats.shape[0]),
        "num_original_texts": len(orig_texts),
        "num_expanded_texts": len(exp_texts),
        "vision_dim": int(vision_feats.shape[1]),
        "text_dim": int(exp_feats.shape[1]),
    })
    print(f"[save] {out_dir}")


def safe_load_tensor(path: Path, key: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    x = obj[key] if isinstance(obj, dict) else obj
    return x.float().cpu()


def stage_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    vision = safe_load_tensor(out_dir / "vision_local_centers.pt", "vision_feats")
    orig = safe_load_tensor(out_dir / "text_original_exact.pt", "text_feats")
    exp = safe_load_tensor(out_dir / "text_expanded_exact.pt", "text_feats")
    m_orig = metrics(vision, orig, args, 0)
    m_exp = metrics(vision, exp, args, 1)
    summary = {
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
    write_json(out_dir / "local_density_report.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[save] {out_dir / 'local_density_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO local-center text/vision density evaluation")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("sample")
    p.add_argument("--coco_root", default="./coco", type=str)
    p.add_argument("--split", default="train2014", type=str)
    p.add_argument("--ann_path", default=None, type=str)
    p.add_argument("--feature_dir", default="feature/coco2014_dinov2_vitb14_448_local_k20", type=str)
    p.add_argument("--num_images", default=1000, type=int)
    p.add_argument("--max_captions", default=5, type=int)
    p.add_argument("--sample_mode", choices=["first", "random"], default="first")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--out_jsonl", default="outputs/local_density_eval/sample_1000.jsonl", type=str)

    p = sub.add_parser("features")
    p.add_argument("--sample_jsonl", default="outputs/local_density_eval/sample_1000.jsonl", type=str)
    p.add_argument("--recaption_jsonl", default=None, type=str)
    p.add_argument("--local_feature_dir", default="feature/coco2014_dinov2_vitb14_448_local_k20", type=str)
    p.add_argument("--atoms_per_image", default=0, type=int, help="0 means infer from local-center feature shape.")
    p.add_argument("--text_model", default="sentence-transformers/all-mpnet-base-v2", type=str)
    p.add_argument("--text_batch_size", default=128, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--out_dir", default="outputs/local_density_eval", type=str)

    p = sub.add_parser("report")
    p.add_argument("--out_dir", default="outputs/local_density_eval", type=str)
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
