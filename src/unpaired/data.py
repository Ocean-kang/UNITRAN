"""ID-level protocols. Paired identities never enter the mapper."""
import json
from collections import defaultdict
from pathlib import Path

import torch

from .config import root_path
from .io import digest, file_digest, manifest, read_jsonl, work_dir, write_json, write_jsonl


def read_coco(root, split, expected=None):
    root = Path(root)
    source = root / "annotations" / f"captions_{split}.json"
    ann = json.loads(source.read_text(encoding="utf-8"))
    images = ann["images"]
    ids = [int(row["id"]) for row in images]
    files = [row["file_name"] for row in images]
    if len(ids) != len(set(ids)) or len(files) != len(set(files)):
        raise ValueError(f"Duplicate image IDs/files in {source}")
    if expected is not None and len(images) != expected:
        raise ValueError(f"{split}: expected {expected} images, found {len(images)}")
    known = set(ids)
    captions, seen_annotations = defaultdict(list), set()
    for row in ann["annotations"]:
        iid, aid = int(row["image_id"]), int(row["id"])
        if iid not in known or aid in seen_annotations:
            raise ValueError(f"Orphan or duplicate caption annotation {aid}")
        text = row["caption"].strip()
        if not text:
            raise ValueError(f"Empty caption {aid}")
        seen_annotations.add(aid)
        captions[iid].append((aid, text))
    rows = []
    for image in sorted(images, key=lambda row: int(row["id"])):
        iid, filename = int(image["id"]), image["file_name"]
        if Path(filename).name != filename or not (root / split / filename).is_file():
            raise ValueError(f"Missing or invalid COCO image: {split}/{filename}")
        # Official COCO filenames encode the annotation ID; reject mismatched metadata.
        if filename.startswith("COCO_") and int(Path(filename).stem.rsplit("_", 1)[1]) != iid:
            raise ValueError(f"Filename/annotation ID mismatch: {filename}")
        if not captions[iid]:
            raise ValueError(f"No captions for image {iid}; refusing to drop a train image")
        entries = sorted(captions[iid])
        rows.append({"sample_id": f"coco2014:{iid}", "image_id": iid,
                     "corpus": "coco2014", "split": split, "file_name": filename,
                     "caption_ids": [a for a, _ in entries], "captions": [t for _, t in entries]})
    return rows


def read_spc(paragraphs_path, metadata_path, coco_ids):
    """Exclude VG images linked to any COCO train/evaluation item, using source provenance."""
    paragraphs = json.loads(Path(paragraphs_path).read_text(encoding="utf-8"))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    by_id = {}
    for row in metadata:
        iid = int(row["image_id"])
        if iid in by_id:
            raise ValueError(f"Duplicate Visual Genome ID {iid}")
        by_id[iid] = row
    rows, excluded, seen = [], [], set()
    for row in paragraphs:
        iid = int(row["image_id"])
        if iid in seen:
            raise ValueError(f"Duplicate SPC image {iid}")
        seen.add(iid)
        origin = by_id.get(iid)
        if origin is None or "coco_id" not in origin:
            raise ValueError(f"Missing COCO provenance for SPC image {iid}")
        coco_id = int(origin["coco_id"]) if origin["coco_id"] is not None else None
        if coco_id in coco_ids:
            excluded.append(iid)
            continue
        text = row["paragraph"].strip()
        if not text:
            raise ValueError(f"Empty SPC paragraph {iid}")
        rows.append({"sample_id": f"spc:{iid}", "image_id": iid, "corpus": "spc",
                     "split": "paragraphs", "coco_id": coco_id, "captions": [text]})
    if not rows:
        raise ValueError("No nonoverlapping SPC paragraphs remain")
    return sorted(rows, key=lambda row: row["image_id"]), excluded


def make_splits(train, evaluation, split, external=None):
    if {r["sample_id"] for r in train} & {r["sample_id"] for r in evaluation}:
        raise ValueError("COCO train2014/val2014 identity overlap")
    g = torch.Generator().manual_seed(int(split["seed"]))
    px = torch.randperm(len(train), generator=g).tolist()
    if split["setting"] == "disjoint":
        ix, iy = px[:len(px) // 2], px[len(px) // 2:]
        text_source = train
    else:
        if not external:
            raise ValueError("cross_dataset requires independently sourced SPC paragraphs")
        ix, iy = px, torch.randperm(len(external), generator=g).tolist()
        text_source = external
    size = split.get("train_size_per_side")
    if size is not None:
        if size > min(len(ix), len(iy)):
            raise ValueError("Requested train size exceeds available samples")
        ix, iy = ix[:size], iy[:size]
    order = torch.randperm(len(evaluation), generator=g).tolist()
    val_size = split.get("val_size")
    if val_size is None:
        val_size = len(order) // 2
    test_size = split.get("test_size")
    if test_size is None:
        test_size = len(order) - val_size
    if min(val_size, test_size) < 2 or val_size + test_size > len(order):
        raise ValueError("Need >=2 paired val and test items, with no overlap")
    # Remove captions entirely from the vision training manifest.
    vx = [{k: v for k, v in train[i].items() if k not in {"captions", "caption_ids"}} for i in ix]
    ty = [{k: v for k, v in text_source[i].items() if k != "file_name"} for i in iy]
    return {"vision_train": vx, "text_train": ty,
            "paired_val": [evaluation[i] for i in order[:val_size]],
            "paired_test": [evaluation[i] for i in order[val_size:val_size + test_size]]}


def prepare(cfg):
    s, root = cfg["split"], root_path(cfg["paths"]["coco_root"])
    train = read_coco(root, "train2014", s.get("expected_train_images"))
    evaluation = read_coco(root, "val2014", s.get("expected_eval_images"))
    sources = {split: file_digest(root / "annotations" / f"captions_{split}.json")
               for split in ("train2014", "val2014")}
    external, excluded = None, []
    if s["setting"] == "cross_dataset":
        p, m = (root_path(cfg["paths"][key]) for key in ("spc_paragraphs", "spc_image_metadata"))
        external, excluded = read_spc(p, m, {r["image_id"] for r in train + evaluation})
        sources.update(spc_paragraphs=file_digest(p), spc_metadata=file_digest(m))
    groups = make_splits(train, evaluation, s, external)
    meta = {"schema": 1, "split_config": s, "sources": sources,
            "counts": {k: len(rows) for k, rows in groups.items()},
            "source_counts": {"train2014": len(train), "val2014": len(evaluation)},
            "excluded_spc_vg_ids": excluded,
            "manifest_digests": {k: digest(rows) for k, rows in groups.items()}}
    meta["split_digest"] = digest(meta)
    out = work_dir(cfg)
    existing = list(out.glob("*.pt")) + list(out.glob("*.jsonl"))
    if existing:
        meta_path = out / "split_meta.json"
        if not meta_path.is_file() or json.loads(meta_path.read_text()) != meta:
            raise ValueError("Existing artifacts differ from requested protocol; choose a new work_dir")
        audit_manifests(cfg)
        return meta
    for name, rows in groups.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    write_json(out / "split_meta.json", meta)
    audit_manifests(cfg)
    return meta


def audit_manifests(cfg):
    meta = json.loads((work_dir(cfg) / "split_meta.json").read_text())
    payload = {k: v for k, v in meta.items() if k != "split_digest"}
    if meta.get("schema") != 1 or meta.get("split_digest") != digest(payload) or meta["split_config"] != cfg["split"]:
        raise ValueError("Protocol metadata/config mismatch")
    groups = {key: manifest(cfg, key) for key in
              ("vision_train", "text_train", "paired_val", "paired_test")}
    for name, rows in groups.items():
        if digest(rows) != meta["manifest_digests"][name] or len(rows) != meta["counts"][name]:
            raise ValueError(f"Manifest content changed: {name}")
        corpus = "spc" if name == "text_train" and cfg["split"]["setting"] == "cross_dataset" else "coco2014"
        for row in rows:
            if row["corpus"] != corpus or row["sample_id"] != f"{corpus}:{row['image_id']}":
                raise ValueError(f"Invalid identity/provenance in {name}")
            if corpus == "coco2014" and row["split"] != ("train2014" if name.endswith("train") else "val2014"):
                raise ValueError(f"Wrong COCO source split in {name}")
            if name == "vision_train" and "captions" in row:
                raise ValueError("Vision training manifest exposes captions")
    sets = {key: {r["sample_id"] for r in rows} for key, rows in groups.items()}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f"Identity leakage between {a} and {b}")
    if cfg["split"]["setting"] == "cross_dataset":
        coco_ids = {r["image_id"] for name, rows in groups.items() if name != "text_train" for r in rows}
        if any(r.get("coco_id") in coco_ids for r in groups["text_train"]):
            raise ValueError("Cross-dataset source-image leakage")
    return meta
