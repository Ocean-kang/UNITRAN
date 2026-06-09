#!/usr/bin/env python3
"""Extract class-level vision/text prototypes from COCO eval images and masks."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.evaluation.text_density import encode_texts, write_json, write_jsonl
from src.features.dinov2 import MODEL_DIMS, extract_patch_tokens, load_dinov2_model

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_image_dir(coco_root: Path, split: str) -> Path:
    for p in [coco_root / split, coco_root / "images" / split]:
        if p.exists():
            return p
    return coco_root / split


def find_ann_path(coco_root: Path, split: str, annotation_type: str) -> Optional[Path]:
    names = []
    if annotation_type in {"auto", "panoptic"}:
        names.append(f"panoptic_{split}.json")
    if annotation_type in {"auto", "instances"}:
        names.append(f"instances_{split}.json")
    for name in names:
        for p in [coco_root / "annotations" / name, coco_root / name]:
            if p.exists():
                return p
    return None


def find_mask_dir(coco_root: Path, split: str) -> Path:
    for p in [coco_root / f"panoptic_{split}", coco_root / "annotations" / f"panoptic_{split}"]:
        if p.exists():
            return p
    return coco_root / f"panoptic_{split}"


def infer_annotation_type(ann_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    name = ann_path.name.lower()
    if name.startswith("panoptic_"):
        return "panoptic"
    if name.startswith("instances_"):
        return "instances"
    first = next(iter(read_json(ann_path).get("annotations", [])), {})
    return "panoptic" if "segments_info" in first else "instances"


def category_rows(ann: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for c in ann.get("categories", []):
        if "id" in c and "name" in c:
            rows.append({"id": int(c["id"]), "name": str(c["name"]), "supercategory": str(c.get("supercategory", ""))})
    if not rows:
        raise ValueError("Annotation file has no categories with id/name.")
    return sorted(rows, key=lambda x: x["id"])


def build_panoptic_rows(ann: Dict[str, Any], image_dir: Path, mask_dir: Path, max_images: Optional[int]) -> List[Dict[str, Any]]:
    images = {int(x["id"]): x for x in ann.get("images", [])}
    rows = []
    for a in ann.get("annotations", []):
        image_id = int(a["image_id"])
        img = images.get(image_id)
        if img is None:
            continue
        rows.append({
            "image_id": image_id,
            "file_name": str(img["file_name"]),
            "image_path": str(image_dir / str(img["file_name"])),
            "mask_path": str(mask_dir / str(a["file_name"])),
            "height": int(img.get("height", -1)),
            "width": int(img.get("width", -1)),
            "segments_info": a.get("segments_info", []),
        })
    rows = sorted(rows, key=lambda x: x["file_name"])
    return rows[:max_images] if max_images and max_images > 0 else rows


def build_instance_rows(ann: Dict[str, Any], image_dir: Path, max_images: Optional[int]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for a in ann.get("annotations", []):
        grouped.setdefault(int(a["image_id"]), []).append(a)
    rows = []
    for img in ann.get("images", []):
        anns = grouped.get(int(img["id"]), [])
        if anns:
            rows.append({
                "image_id": int(img["id"]),
                "file_name": str(img["file_name"]),
                "image_path": str(image_dir / str(img["file_name"])),
                "height": int(img.get("height", -1)),
                "width": int(img.get("width", -1)),
                "annotations": anns,
            })
    rows = sorted(rows, key=lambda x: x["file_name"])
    return rows[:max_images] if max_images and max_images > 0 else rows


def rgb_to_id(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int64)
    return rgb[:, :, 0] + 256 * rgb[:, :, 1] + 256 * 256 * rgb[:, :, 2]


def panoptic_label_map(row: Dict[str, Any]) -> np.ndarray:
    seg = rgb_to_id(np.asarray(Image.open(row["mask_path"]).convert("RGB")))
    label = np.full(seg.shape, -1, dtype=np.int32)
    for s in row["segments_info"]:
        label[seg == int(s["id"])] = int(s["category_id"])
    return label


def decode_uncompressed_rle(segmentation: Dict[str, Any]) -> Optional[np.ndarray]:
    counts = segmentation.get("counts")
    size = segmentation.get("size")
    if not isinstance(counts, list) or not isinstance(size, list) or len(size) != 2:
        return None
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=np.uint8)
    idx, value = 0, 0
    for run in counts:
        run = int(run)
        if value == 1:
            flat[idx: idx + run] = 1
        idx += run
        value = 1 - value
    return flat.reshape((w, h)).T


def annotation_mask(annotation: Dict[str, Any], height: int, width: int) -> Optional[np.ndarray]:
    seg = annotation.get("segmentation")
    if isinstance(seg, list):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for poly in seg:
            if len(poly) >= 6:
                draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], outline=1, fill=1)
        return np.asarray(mask, dtype=np.uint8)
    if isinstance(seg, dict):
        return decode_uncompressed_rle(seg)
    return None


def instances_label_map(row: Dict[str, Any]) -> np.ndarray:
    height, width = int(row["height"]), int(row["width"])
    label = np.full((height, width), -1, dtype=np.int32)
    for a in sorted(row["annotations"], key=lambda x: float(x.get("area", 0)), reverse=True):
        mask = annotation_mask(a, height, width)
        if mask is not None:
            label[mask > 0] = int(a["category_id"])
    return label


def resize_label(label: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    encoded = (label.astype(np.int64) + 1).clip(0, 65535).astype(np.uint16)
    resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
    return np.asarray(Image.fromarray(encoded).resize(size, resample=resample), dtype=np.int64) - 1


def labels_to_patch_grid(label: np.ndarray, image_size: int, patch_size: int, mode: str) -> np.ndarray:
    grid = image_size // patch_size
    if mode == "nearest":
        return resize_label(label, (grid, grid)).reshape(-1)
    resized = resize_label(label, (image_size, image_size))
    patches = resized.reshape(grid, patch_size, grid, patch_size).transpose(0, 2, 1, 3).reshape(grid * grid, -1)
    out = np.full((grid * grid,), -1, dtype=np.int64)
    for i, values in enumerate(patches):
        values = values[values >= 0]
        if len(values) == 0:
            continue
        ids, counts = np.unique(values, return_counts=True)
        out[i] = ids[counts.argmax()]
    return out


class EvalPrototypeDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], annotation_type: str, image_size: int, patch_size: int, mask_to_patch: str):
        self.rows = list(rows)
        self.annotation_type = annotation_type
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_to_patch = mask_to_patch
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def transform_image(self, path: str) -> torch.Tensor:
        resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        with Image.open(path) as im:
            im = im.convert("RGB").resize((self.image_size, self.image_size), resample=resample)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return (torch.from_numpy(arr).permute(2, 0, 1).contiguous() - self.mean) / self.std

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        try:
            image = self.transform_image(row["image_path"])
            label = panoptic_label_map(row) if self.annotation_type == "panoptic" else instances_label_map(row)
            patch_labels = labels_to_patch_grid(label, self.image_size, self.patch_size, self.mask_to_patch)
            ok, err = True, ""
        except Exception as e:
            grid = self.image_size // self.patch_size
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            patch_labels = np.full((grid * grid,), -1, dtype=np.int64)
            ok, err = False, repr(e)
        return {"image": image, "patch_labels": torch.from_numpy(patch_labels).long(), "image_id": int(row["image_id"]), "file_name": row["file_name"], "ok": ok, "error": err}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "patch_labels": torch.stack([b["patch_labels"] for b in batch], dim=0),
        "image_ids": [int(b["image_id"]) for b in batch],
        "file_names": [str(b["file_name"]) for b in batch],
        "ok": [bool(b["ok"]) for b in batch],
        "errors": [str(b["error"]) for b in batch],
    }


def add_features(sums: Dict[int, torch.Tensor], counts: Dict[int, int], feats: torch.Tensor, labels: torch.Tensor) -> None:
    for cid in labels.unique().tolist():
        cid = int(cid)
        if cid < 0 or cid not in sums:
            continue
        mask = labels == cid
        if mask.any():
            sums[cid] += feats[mask].double().sum(dim=0)
            counts[cid] += int(mask.sum().item())


def load_rows(args: argparse.Namespace) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Path, Path, Optional[Path]]:
    coco_root = Path(args.coco_root)
    image_dir = Path(args.image_dir) if args.image_dir else find_image_dir(coco_root, args.split)
    ann_path = Path(args.ann_path) if args.ann_path else find_ann_path(coco_root, args.split, args.annotation_type)
    if ann_path is None or not ann_path.exists():
        raise FileNotFoundError(f"Cannot find annotation json for split={args.split} under {coco_root}")
    ann = read_json(ann_path)
    annotation_type = infer_annotation_type(ann_path, args.annotation_type)
    categories = category_rows(ann)
    mask_dir = None
    if annotation_type == "panoptic":
        mask_dir = Path(args.mask_dir) if args.mask_dir else find_mask_dir(coco_root, args.split)
        rows = build_panoptic_rows(ann, image_dir, mask_dir, args.max_images)
    else:
        rows = build_instance_rows(ann, image_dir, args.max_images)
    if not rows:
        raise ValueError(f"No usable images found in {ann_path}")
    return annotation_type, categories, rows, image_dir, ann_path, mask_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract class-level eval prototypes from COCO images and masks.")
    parser.add_argument("--coco_root", type=str, default="data/mscoco17")
    parser.add_argument("--split", type=str, default="val2017")
    parser.add_argument("--annotation_type", type=str, default="auto", choices=["auto", "panoptic", "instances"])
    parser.add_argument("--ann_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="feature/eval/mscoco2017")
    parser.add_argument("--out_name", type=str, default="mscoco2017_eval_prototypes.pt")
    parser.add_argument("--model", type=str, default="dinov2_vitb14", choices=sorted(MODEL_DIMS.keys()))
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--mask_to_patch", type=str, default="majority", choices=["majority", "nearest"])
    parser.add_argument("--text_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--text_batch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--min_patch_count", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16", "none"])
    parser.add_argument("--normalize_vision", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.image_size % args.patch_size != 0:
        raise ValueError(f"image_size must be divisible by patch_size, got {args.image_size}/{args.patch_size}")

    out_dir = Path(args.out_dir)
    out_path = out_dir / args.out_name
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Use --overwrite to rerun.")

    annotation_type, categories, rows, image_dir, ann_path, mask_dir = load_rows(args)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = load_dinov2_model(args.model, device=device)

    dataset = EvalPrototypeDataset(rows, annotation_type, args.image_size, args.patch_size, args.mask_to_patch)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=False, collate_fn=collate_fn)

    feature_dim = MODEL_DIMS[args.model]
    cat_ids = [int(c["id"]) for c in categories]
    sums = {cid: torch.zeros(feature_dim, dtype=torch.float64) for cid in cat_ids}
    counts = {cid: 0 for cid in cat_ids}
    failed = []
    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    start = time.time()

    for batch in tqdm(loader, desc="extract eval prototypes"):
        images = batch["images"].to(device, non_blocking=True)
        patch_tokens = extract_patch_tokens(model, images, use_amp=use_amp, amp_dtype=amp_dtype).detach().cpu().float()
        labels = batch["patch_labels"]
        for i in range(len(batch["file_names"])):
            if not batch["ok"][i]:
                failed.append({"image_id": batch["image_ids"][i], "file_name": batch["file_names"][i], "error": batch["errors"][i]})
                continue
            add_features(sums, counts, patch_tokens[i], labels[i])

    valid_categories = [c for c in categories if counts[int(c["id"])] >= args.min_patch_count]
    if not valid_categories:
        raise RuntimeError("No class has enough masked patches. Check annotation paths and mask format.")

    vision = torch.stack([(sums[int(c["id"])] / counts[int(c["id"])]).float() for c in valid_categories], dim=0)
    if args.normalize_vision:
        vision = F.normalize(vision, dim=-1)

    class_ids = [int(c["id"]) for c in valid_categories]
    class_names = [str(c["name"]).replace("_", " ") for c in valid_categories]
    supercategories = [str(c.get("supercategory", "")) for c in valid_categories]
    text = encode_texts(class_names, args.text_model, args.text_batch_size, str(device))

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "vision_feats": vision,
        "text_feats": text,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "class_names": class_names,
        "text_prompts": class_names,
        "supercategories": supercategories,
        "vision_patch_counts": torch.tensor([counts[cid] for cid in class_ids], dtype=torch.long),
        "source_key": "vision_feats",
        "target_key": "text_feats",
        "annotation_type": annotation_type,
        "dataset_root": str(Path(args.coco_root)),
        "split": args.split,
        "model": args.model,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "mask_to_patch": args.mask_to_patch,
        "text_model": args.text_model,
    }
    torch.save(payload, out_path)

    stem = out_path.stem
    meta_rows = []
    for i, c in enumerate(valid_categories):
        cid = int(c["id"])
        meta_rows.append({
            "row": i,
            "class_id": cid,
            "class_name": str(c["name"]).replace("_", " "),
            "supercategory": str(c.get("supercategory", "")),
            "vision_patch_count": counts[cid],
        })
    write_jsonl(out_dir / f"{stem}_meta.jsonl", meta_rows)
    write_json(out_dir / f"{stem}_config.json", {
        "out_path": str(out_path),
        "num_classes": len(valid_categories),
        "num_all_categories": len(categories),
        "num_images": len(rows),
        "num_failed_images": len(failed),
        "failed_images": failed[:50],
        "coco_root": args.coco_root,
        "split": args.split,
        "annotation_type": annotation_type,
        "ann_path": str(ann_path),
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir) if mask_dir else None,
        "model": args.model,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "mask_to_patch": args.mask_to_patch,
        "min_patch_count": args.min_patch_count,
        "normalize_vision": bool(args.normalize_vision),
        "text_model": args.text_model,
        "elapsed_sec": round(time.time() - start, 3),
    })
    print(f"[save] {out_path}")
    print(f"[classes] valid={len(valid_categories)}, all={len(categories)}")
    print(f"[failed] {len(failed)}")


if __name__ == "__main__":
    main()
