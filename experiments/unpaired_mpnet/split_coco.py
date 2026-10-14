"""Rosetta-style image-ID-disjoint MSCOCO train2014 splitting + separate paired val2014."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from common import coco_root, data_dir, dump_json, load_config, read_jsonl, write_jsonl


def read_coco(root, split: str) -> list[dict]:
    ann_path = root / "annotations" / f"captions_{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(ann_path)
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    captions = defaultdict(list)
    for r in ann["annotations"]:
        captions[int(r["image_id"])].append(str(r["caption"]))
    rows = []
    for img in sorted(ann["images"], key=lambda x: int(x["id"])):
        iid = int(img["id"])
        texts = [t.strip() for t in captions[iid] if t.strip()]
        if not texts:
            continue
        path = root / split / img["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing COCO image {path}")
        rows.append({"image_id": iid, "file_name": img["file_name"], "split": split, "captions": texts})
    if not rows:
        raise RuntimeError(f"No captioned images in {ann_path}")
    return rows


def make_splits(train_rows: list[dict], val_rows: list[dict], seed: int,
                train_size_per_side: int | None, val_size: int | None) -> dict[str, list[dict]]:
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(len(train_rows), generator=g).tolist()
    half = len(p) // 2
    xa, ya = p[:half], p[half:]
    if train_size_per_side is not None:
        if train_size_per_side < 1 or train_size_per_side > len(xa):
            raise ValueError(f"train_size_per_side must be in [1,{len(xa)}]")
        xa, ya = xa[:train_size_per_side], ya[:train_size_per_side]
    # A second deterministic draw is used for paired val2014 subsampling.
    v = torch.randperm(len(val_rows), generator=g).tolist()
    if val_size is not None:
        if val_size < 2 or val_size > len(v):
            raise ValueError(f"val_size must be in [2,{len(v)}]")
        v = v[:val_size]
    out = {"vision_train": [train_rows[i] for i in xa],
           "text_train": [train_rows[i] for i in ya],
           "paired_val": [val_rows[i] for i in v]}
    ids = {k: {int(r["image_id"]) for r in rows} for k, rows in out.items()}
    if ids["vision_train"] & ids["text_train"]:
        raise AssertionError("Overlapping unpaired training groups")
    if ids["paired_val"] & (ids["vision_train"] | ids["text_train"]):
        raise AssertionError("Train/val IDs overlap")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiments/unpaired_mpnet/config.yaml")
    p.add_argument("--force", action="store_true", help="Replace existing manifests (not .pt banks)")
    args = p.parse_args()
    cfg = load_config(args.config)
    root, out = coco_root(cfg), data_dir(cfg)
    tr = read_coco(root, "train2014")
    va = read_coco(root, "val2014")
    s = cfg["split"]
    groups = make_splits(tr, va, int(s["seed"]), s.get("train_size_per_side"), s.get("val_size"))
    existing = [out / f"{key}.jsonl" for key in groups if (out / f"{key}.jsonl").exists()]
    existing_banks = [out / f"{n}.pt" for n in ("vision_train", "text_train", "vision_val", "text_val") if (out / f"{n}.pt").exists()]
    if existing_banks and args.force:
        raise RuntimeError("Cannot replace manifests while feature banks exist; choose a new paths.work_dir or remove old banks explicitly")
    if existing and not args.force:
        same = len(existing) == len(groups) and all(read_jsonl(out / f"{k}.jsonl") == rows for k, rows in groups.items())
        if not same:
            raise FileExistsError(f"Inconsistent existing manifests in {out}. Use a new work_dir or --force (no banks).")
        print(f"[skip] matching manifests already exist at {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    for key, rows in groups.items():
        write_jsonl(out / f"{key}.jsonl", rows)
        print(f"[split] {key}: {len(rows)}")
    dump_json(out / "split_meta.json", {
        "source_train": str(root / "annotations/captions_train2014.json"),
        "source_val": str(root / "annotations/captions_val2014.json"),
        "seed": s["seed"],
        "train_size_per_side": s.get("train_size_per_side"),
        "val_size": s.get("val_size"),
        "counts": {key: len(rows) for key, rows in groups.items()},
        "method": "Rosetta disjoint_split: one seeded permutation, two nonoverlapping halves",
    })
    print(f"[save] manifests in {out}")


if __name__ == "__main__":
    main()
