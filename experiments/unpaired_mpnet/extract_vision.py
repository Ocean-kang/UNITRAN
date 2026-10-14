"""Image-level DINOv2 embeddings for the Rosetta-style UNITRAN experiment."""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import coco_root, load_config, manifest, save_bank, validate_cache
from src.features.dinov2 import CocoImageDataset, collate_fn, load_dinov2_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiments/unpaired_mpnet/config.yaml")
    p.add_argument("--split", choices=["vision_train", "paired_val"], required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    vcfg = cfg["vision"]
    rows = manifest(cfg, args.split)
    target = "vision_train" if args.split == "vision_train" else "vision_val"
    model_name, image_size, pooling = vcfg["model"], int(vcfg["image_size"]), vcfg["pooling"]
    model_label = f"{model_name}@{image_size}:{pooling}"
    if not args.force and validate_cache(cfg, target, rows, model_label):
        return
    data = CocoImageDataset([
        {"index": i, "path": str(coco_root(cfg) / r["split"] / r["file_name"])}
        for i, r in enumerate(rows)
    ], image_size)
    device = torch.device(cfg["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; change runtime.device to cpu for debugging")
    loader = DataLoader(data, batch_size=int(vcfg["batch_size"]), shuffle=False,
                        num_workers=int(vcfg["num_workers"]), collate_fn=collate_fn, pin_memory=(device.type == "cuda"))
    model = load_dinov2_model(model_name, device)
    features = []
    for batch in tqdm(loader, desc=f"DINOv2 {args.split}"):
        if not all(batch["ok"]):
            errors = [str(rows[i]["image_id"]) + ": " + e for i, ok, e in zip(batch["local_indices"], batch["ok"], batch["errors"]) if not ok]
            raise RuntimeError("Failed to decode image: " + "; ".join(errors[:4]))
        x = batch["images"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            out = model.forward_features(x)
            patch = out["x_norm_patchtokens"]
            if pooling == "patch_mean":
                pooled = patch.mean(dim=1)
            elif pooling == "all_token_mean":
                pieces = [out["x_norm_clstoken"].unsqueeze(1)]
                if "x_norm_regtokens" in out:
                    pieces.append(out["x_norm_regtokens"])
                pieces.append(patch)
                pooled = torch.cat(pieces, dim=1).mean(dim=1)
            else:
                raise ValueError(f"Unknown pooling={pooling}")
            pooled = F.normalize(pooled.float(), dim=-1).cpu()
        features.append(pooled)
    tensor = torch.cat(features, dim=0)
    save_bank(cfg, target, rows, tensor, model_label, force=args.force)


if __name__ == "__main__":
    main()
