"""Atomic artifact IO and content-based cache identity."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from .config import root_path

SCHEMA = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def save_tensor(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def work_dir(cfg):
    return root_path(cfg["paths"]["work_dir"])


def manifest(cfg, name):
    rows = read_jsonl(work_dir(cfg) / f"{name}.jsonl")
    ids = [r["sample_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate identities in {name}")
    return rows


def bank_manifest(name):
    modality, split = name.split("_", 1)
    return name if split == "train" else f"paired_{split}"


def encoder_spec(cfg, modality):
    if modality == "vision":
        spec = {key: cfg["vision"][key] for key in
                ("model", "image_size", "pooling", "repo", "weights", "revision")}
        repo = spec["repo"] or os.environ.get("DINOV2_REPO")
        if repo:
            repo = root_path(repo)
            spec["repo"] = str(repo.resolve())
            spec["local_source_digest"] = digest({str(p.relative_to(repo)): file_digest(p)
                                                 for p in sorted(repo.rglob("*.py"))})
            if spec["revision"]:
                actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
                if actual != spec["revision"]:
                    raise ValueError("Local DINOv2 revision differs from config")
        if spec["weights"]:
            spec["weights_sha256"] = file_digest(root_path(spec["weights"]))
        return spec
    spec = {"model": cfg["text"]["model"], "revision": cfg["text"]["revision"],
            "pooling": "normalized_caption_mean"}
    directory = root_path(spec["model"])
    if directory.is_dir():
        spec["local_model_digest"] = digest({str(p.relative_to(directory)): file_digest(p)
                                            for p in sorted(directory.rglob("*")) if p.is_file()})
    return spec


def bank_metadata(cfg, name, rows):
    return {"schema": SCHEMA, "name": name, "manifest_digest": digest(rows),
            "encoder": encoder_spec(cfg, name.split("_")[0]),
            "split_digest": json.loads((work_dir(cfg) / "split_meta.json").read_text())["split_digest"]}


def validate_features(features, ids):
    if not torch.is_tensor(features) or features.ndim != 2 or len(features) != len(ids) or features.shape[1] < 1:
        raise ValueError("Invalid feature bank shape")
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Empty or duplicate bank identities")
    if not torch.isfinite(features).all() or (features.norm(dim=1) < 1e-12).any():
        raise ValueError("Nonfinite or zero feature vectors")


def load_bank(cfg, name):
    rows = manifest(cfg, bank_manifest(name))
    obj = torch.load(work_dir(cfg) / f"{name}.pt", map_location="cpu", weights_only=True)
    if obj.get("metadata") != bank_metadata(cfg, name, rows):
        raise ValueError(f"Stale/incompatible cache: {name}; extract into a new work_dir")
    ids = [r["sample_id"] for r in rows]
    if obj.get("sample_ids") != ids:
        raise ValueError(f"Bank/manifest order mismatch: {name}")
    validate_features(obj["features"], ids)
    return obj


def save_bank(cfg, name, rows, features):
    ids = [r["sample_id"] for r in rows]
    validate_features(features, ids)
    save_tensor(work_dir(cfg) / f"{name}.pt", {
        "features": features.float().cpu().contiguous(), "sample_ids": ids,
        "metadata": bank_metadata(cfg, name, rows),
    })
