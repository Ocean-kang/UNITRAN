"""Encode existing COCO captions using MPNet (no Llama or recaption generation)."""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from common import load_config, manifest, save_bank, validate_cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiments/unpaired_mpnet/config.yaml")
    p.add_argument("--split", choices=["text_train", "paired_val"], required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    rows = manifest(cfg, args.split)
    target = "text_train" if args.split == "text_train" else "text_val"
    tcfg = cfg["text"]
    model_id = tcfg["model"]
    if not args.force and validate_cache(cfg, target, rows, f"{model_id}:caption_mean"):
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=cfg["runtime"]["device"])
    output = []
    batch_images = int(tcfg["image_batch_size"])
    for start in tqdm(range(0, len(rows), batch_images), desc=f"MPNet {args.split}"):
        chunk = rows[start:start+batch_images]
        captions = []
        sizes = []
        for row in chunk:
            texts = [str(t).strip() for t in row["captions"] if str(t).strip()]
            if not texts:
                raise ValueError(f"No captions for image_id={row['image_id']}")
            captions.extend(texts)
            sizes.append(len(texts))
        emb = torch.from_numpy(model.encode(captions,
            batch_size=int(tcfg["text_batch_size"]), show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True)).float()
        parts, offset = [], 0
        for size in sizes:
            # Equivalent to Rosetta reduce_texts: normalized captions -> mean -> re-normalize.
            parts.append(F.normalize(emb[offset:offset+size].mean(dim=0), dim=-1))
            offset += size
        output.append(torch.stack(parts))
    feats = torch.cat(output, dim=0)
    save_bank(cfg, target, rows, feats, f"{model_id}:caption_mean", force=args.force)


if __name__ == "__main__":
    main()
