"""Frozen encoders, one vector per image/paragraph; extraction preserves manifest order."""
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import root_path
from .data import audit_manifests
from .io import bank_manifest, load_bank, manifest, save_bank, work_dir


def pool_vision(output, pooling):
    patch = output["x_norm_patchtokens"]
    if pooling == "patch_mean":
        pooled = patch.mean(dim=1)
    elif pooling == "all_token_mean":
        pieces = [output["x_norm_clstoken"].unsqueeze(1)]
        if "x_norm_regtokens" in output:
            pieces.append(output["x_norm_regtokens"])
        pooled = torch.cat(pieces + [patch], dim=1).mean(dim=1)
    elif pooling == "cls":
        pooled = output["x_norm_clstoken"]
    else:
        raise ValueError(f"Unknown pooling {pooling}")
    return F.normalize(pooled.float(), dim=-1)


def caption_means(embeddings, sizes):
    if min(sizes) < 1 or sum(sizes) != len(embeddings):
        raise ValueError("Caption counts do not match encoded rows")
    embeddings = F.normalize(embeddings.float(), dim=-1)
    return F.normalize(torch.stack([part.mean(dim=0) for part in embeddings.split(sizes)]), dim=-1)


def load_vision(cfg, device):
    # Reuse dataset preprocessing, but resolve the encoder revision explicitly.
    v = cfg["vision"]
    repo = v.get("repo") or os.environ.get("DINOV2_REPO")
    if repo:
        repo = root_path(repo)
        if v.get("revision"):
            actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
            if actual != v["revision"] or dirty:
                raise ValueError("Local DINOv2 repo must be clean and match vision.revision")
        model = torch.hub.load(str(repo), v["model"], source="local", pretrained=not bool(v.get("weights")))
    else:
        if not v.get("revision"):
            raise ValueError("Pin vision.revision or provide a local vision.repo")
        model = torch.hub.load(f"facebookresearch/dinov2:{v['revision']}", v["model"],
                               trust_repo=True, skip_validation=True, pretrained=not bool(v.get("weights")))
    if v.get("weights"):
        state = torch.load(root_path(v["weights"]), map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("state_dict", state), strict=True)
    return model.eval().to(device)


def extract(cfg, name, force=False):
    audit_manifests(cfg)
    rows = manifest(cfg, bank_manifest(name))
    if (work_dir(cfg) / f"{name}.pt").exists() and not force:
        load_bank(cfg, name)
        print(f"[cache] {name}")
        return
    modality = name.split("_")[0]
    device = torch.device(cfg["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select runtime.device=cpu")
    features = []
    if modality == "vision":
        from src.features.dinov2 import CocoImageDataset, collate_fn
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        v = cfg["vision"]
        root = root_path(cfg["paths"]["coco_root"])
        data = CocoImageDataset([{"index": i, "path": str(root / r["split"] / r["file_name"])}
                                 for i, r in enumerate(rows)], v["image_size"])
        loader = DataLoader(data, batch_size=v["batch_size"], shuffle=False, num_workers=v["num_workers"],
                            collate_fn=collate_fn, pin_memory=device.type == "cuda")
        model = load_vision(cfg, device)
        for batch in tqdm(loader, desc=name):
            if not all(batch["ok"]):
                raise RuntimeError(f"Image decoding failed: {batch['errors']}")
            with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
                output = model.forward_features(batch["images"].to(device))
                features.append(pool_vision(output, v["pooling"]).cpu())
    else:
        from sentence_transformers import SentenceTransformer
        t = cfg["text"]
        model_path = root_path(t["model"])
        model_id = str(model_path) if model_path.is_dir() else t["model"]
        model = SentenceTransformer(model_id, revision=t.get("revision"), device=str(device))
        for start in tqdm(range(0, len(rows), t["image_batch_size"]), desc=name):
            chunk = rows[start:start + t["image_batch_size"]]
            captions = [caption for row in chunk for caption in row["captions"]]
            emb = model.encode(captions, batch_size=t["text_batch_size"], show_progress_bar=False,
                               convert_to_tensor=True, normalize_embeddings=True)
            features.append(caption_means(emb.cpu(), [len(row["captions"]) for row in chunk]))
    save_bank(cfg, name, rows, torch.cat(features))
