"""Proper ImageID-paired val2014 retrieval; FOSCTTM, mean rank, I2T/T2I Recall@K."""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from common import assert_disjoint_and_paired, dump_json, load_bank, load_config, root_path
from metrics import retrieval


def norm_center(train: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    # UNITRAN's load_mapping_data separately centers the validation set.
    return F.normalize(val.float() - val.float().mean(dim=0, keepdim=True), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/unpaired_mpnet/config.yaml")
    parser.add_argument("--weight", default=None, help="Optional override for mapping W checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    assert_disjoint_and_paired(cfg)
    vx, vy = load_bank(cfg, "vision_val"), load_bank(cfg, "text_val")
    xe = norm_center(load_bank(cfg, "vision_train")["features"], vx["features"])
    ye = norm_center(load_bank(cfg, "text_train")["features"], vy["features"])
    out = root_path(cfg["paths"]["result_dir"])
    seed = int(cfg["mapping"]["seed"])
    weight_path = root_path(args.weight) if args.weight else out / f"W_mini_vec2vec_vision_to_text_seed{seed}.pt"
    W = torch.load(weight_path, map_location="cpu", weights_only=True).float()
    if W.shape != (xe.shape[1], ye.shape[1]):
        raise ValueError(f"W has incompatible shape {tuple(W.shape)}")
    mapped = F.normalize(xe @ W, dim=-1)
    evaluation = cfg["evaluation"]
    bs, device = int(evaluation["batch_size"]), str(evaluation["device"])
    result = {
        "model": "UNITRAN mini_vec2vec", "weight": str(weight_path),
        "num_validation_pairs": len(xe),
        "i2t": retrieval(mapped, ye, block_size=bs, device=device),
        # Reverse-direction ranking uses the transpose of the same similarity function.
        "t2i": retrieval(ye, mapped, block_size=bs, device=device),
        "note": "Validation uses true image_id-aligned val2014 pairs. No paired training data.",
    }
    out_file = out / f"retrieval_mini_vec2vec_seed{seed}.json"
    dump_json(out_file, result)
    print(f"[I2T] {result['i2t']}")
    print(f"[T2I] {result['t2i']}")
    print(f"[save] {out_file}")


if __name__ == "__main__":
    main()
