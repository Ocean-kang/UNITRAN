#!/usr/bin/env python3
"""Query MS COCO captions by image file name."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_coco_image_id(file_name: str) -> Optional[int]:
    match = re.search(r"_(\d{12})(?:\.|$)", Path(file_name).name)
    if match:
        return int(match.group(1))

    numbers = re.findall(r"\d+", Path(file_name).stem)
    return int(numbers[-1]) if numbers else None


def find_default_ann(coco_root: Path, split: str) -> Path:
    candidates = [
        coco_root / "annotations" / f"captions_{split}.json",
        coco_root / f"captions_{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find captions_{split}.json under {coco_root}")


def load_caption_index(ann_path: Path) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    with ann_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)

    file_name_to_id: Dict[str, int] = {}
    for image in ann.get("images", []):
        if "file_name" not in image or "id" not in image:
            continue
        file_name_to_id[Path(str(image["file_name"])).name] = int(image["id"])

    captions_by_image_id: Dict[int, List[str]] = {}
    for item in ann.get("annotations", []):
        if "image_id" not in item or "caption" not in item:
            continue
        image_id = int(item["image_id"])
        captions_by_image_id.setdefault(image_id, []).append(str(item["caption"]))

    if not captions_by_image_id:
        raise ValueError(f"No captions found in annotation file: {ann_path}")

    return file_name_to_id, captions_by_image_id


def query_captions(image_name: str, ann_path: Path) -> Tuple[int, List[str]]:
    file_name_to_id, captions_by_image_id = load_caption_index(ann_path)
    base_name = Path(image_name).name
    image_id = file_name_to_id.get(base_name)

    if image_id is None:
        image_id = parse_coco_image_id(base_name)
    if image_id is None:
        raise ValueError(f"Cannot parse image id from image name: {image_name}")

    captions = captions_by_image_id.get(image_id, [])
    if not captions:
        raise KeyError(f"No captions found for image_name={image_name}, image_id={image_id}")

    return image_id, captions


def main() -> None:
    parser = argparse.ArgumentParser(description="Query MS COCO captions by image file name.")
    parser.add_argument("--image_name", type=str, required=True, help="Example: COCO_train2014_000000000009.jpg")
    parser.add_argument("--coco_root", type=str, default="./coco", help="COCO root, default: ./coco")
    parser.add_argument("--split", type=str, default="train2014", help="COCO split, default: train2014")
    parser.add_argument("--ann_path", type=str, default=None, help="Optional captions annotation path")
    args = parser.parse_args()

    ann_path = Path(args.ann_path) if args.ann_path else find_default_ann(Path(args.coco_root), args.split)
    image_id, captions = query_captions(args.image_name, ann_path)

    print(f"image_name: {Path(args.image_name).name}")
    print(f"image_id: {image_id}")
    print(f"num_captions: {len(captions)}")
    for i, caption in enumerate(captions, start=1):
        print(f"{i}. {caption}")


if __name__ == "__main__":
    main()
