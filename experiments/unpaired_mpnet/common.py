"""Shared, experiment-local IO helpers. No changes to UNITRAN src/ are required."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML dict at {path}")
    return cfg


def root_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def data_dir(cfg: dict) -> Path:
    return root_path(cfg["paths"]["work_dir"])


def coco_root(cfg: dict) -> Path:
    return root_path(cfg["paths"]["coco_root"])


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest(cfg: dict, name: str) -> list[dict]:
    path = data_dir(cfg) / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run split_coco.py first")
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty manifest {path}")
    ids = [int(r["image_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate image_id found in {path}")
    return rows


def feature_path(cfg: dict, name: str) -> Path:
    return data_dir(cfg) / f"{name}.pt"


def load_bank(cfg: dict, name: str) -> dict:
    p = feature_path(cfg, name)
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}; run feature extraction first")
    obj = torch.load(p, map_location="cpu", weights_only=True)
    feats, ids = obj["features"], obj["image_ids"]
    if feats.ndim != 2 or ids.ndim != 1 or len(feats) != len(ids):
        raise ValueError(f"Invalid bank shape: {p}")
    if not torch.isfinite(feats).all():
        raise ValueError(f"NaN/Inf in {p}")
    return obj


def save_bank(cfg: dict, name: str, rows: list[dict], feats: torch.Tensor, model: str, *, force: bool = False) -> None:
    p = feature_path(cfg, name)
    expected = torch.tensor([int(r["image_id"]) for r in rows], dtype=torch.long)
    if feats.ndim != 2 or len(feats) != len(expected):
        raise ValueError(f"Invalid features shape {tuple(feats.shape)} for {len(expected)} ids")
    if not torch.isfinite(feats).all():
        raise ValueError("Features contain NaN/Inf")
    if p.exists() and not force:
        raise FileExistsError(f"{p} exists; pass --force to regenerate")
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feats.float().contiguous(), "image_ids": expected, "model": model}, p)
    print(f"[save] {p}: {tuple(feats.shape)}")


def assert_disjoint_and_paired(cfg: dict) -> None:
    vx = load_bank(cfg, "vision_train")["image_ids"].tolist()
    ty = load_bank(cfg, "text_train")["image_ids"].tolist()
    ve = load_bank(cfg, "vision_val")["image_ids"].tolist()
    te = load_bank(cfg, "text_val")["image_ids"].tolist()
    if set(vx) & set(ty):
        raise ValueError("Vision and text training ID sets overlap; not strictly unpaired")
    if ve != te:
        raise ValueError("Validation banks must have identical image_id order")
    if (set(vx) | set(ty)) & set(ve):
        raise ValueError("Training and validation image_id sets overlap")


def validate_cache(cfg: dict, name: str, rows: list[dict], model: str) -> bool:
    p = feature_path(cfg, name)
    if not p.exists():
        return False
    obj = load_bank(cfg, name)
    ids = [int(r["image_id"]) for r in rows]
    if obj["image_ids"].tolist() != ids or obj.get("model") != model:
        raise ValueError(f"Existing bank {p} has different IDs/model; remove it or use --force")
    print(f"[skip] already extracted {p}")
    return True
