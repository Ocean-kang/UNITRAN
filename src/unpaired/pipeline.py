"""Orchestration keeps training inputs and paired evaluation inputs separate."""
import argparse
import importlib.metadata
import json
import subprocess
import time

import torch
from threadpoolctl import threadpool_limits

from src.evaluation.retrieval import retrieval
from src.training.mini_vec2vec import fit_unpaired, preprocess
from .config import ROOT, load_config, root_path
from .data import audit_manifests, prepare
from .features import extract
from .io import digest, file_digest, load_bank, save_tensor, work_dir, write_json


def environment():
    versions = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "sentence-transformers", "transformers", "Pillow", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", "HEAD"]).hex()
    except (OSError, subprocess.CalledProcessError):
        commit, diff = None, ""
    return {"packages": versions, "git_commit": commit, "git_diff_digest": digest(diff),
            "cuda_available": torch.cuda.is_available()}


def training_identity(cfg):
    return {name: file_digest(work_dir(cfg) / f"{name}.pt") for name in ("vision_train", "text_train")}


def train(cfg, force=False):
    meta = audit_manifests(cfg)
    # Never open vision_val/text_val/vision_test/text_test during fitting.
    vx, ty = load_bank(cfg, "vision_train"), load_bank(cfg, "text_train")
    out = root_path(cfg["paths"]["result_dir"])
    weight_path = out / "mapping.pt"
    if weight_path.exists() and not force:
        raise FileExistsError(f"{weight_path} exists; use --force to rerun explicitly")
    start = time.monotonic()
    print("[fit] vision -> text")
    forward, h_forward = fit_unpaired(vx["features"], ty["features"], cfg["mapping"])
    print("[fit] text -> vision")
    reverse, h_reverse = fit_unpaired(ty["features"], vx["features"], cfg["mapping"])
    checkpoint = {"schema": 1, "forward": forward, "reverse": reverse,
                  "split_digest": meta["split_digest"], "training_banks": training_identity(cfg),
                  "encoders": {"vision": vx["metadata"]["encoder"], "text": ty["metadata"]["encoder"]},
                  "config": cfg, "environment": environment()}
    save_tensor(weight_path, checkpoint)
    write_json(out / "resolved_config.json", cfg)
    write_json(out / "training.json", {"forward": h_forward, "reverse": h_reverse,
               "setting": cfg["split"]["setting"], "seconds": time.monotonic() - start,
               "environment": checkpoint["environment"], "paired_training_samples": 0})
    return checkpoint


def evaluate(cfg, split="val"):
    meta = audit_manifests(cfg)
    out = root_path(cfg["paths"]["result_dir"])
    checkpoint = torch.load(out / "mapping.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != 1 or checkpoint["split_digest"] != meta["split_digest"]:
        raise ValueError("Checkpoint belongs to another split/protocol")
    if checkpoint["training_banks"] != training_identity(cfg) or checkpoint["config"]["mapping"] != cfg["mapping"]:
        raise ValueError("Checkpoint/training cache or mapping configuration mismatch")
    vx, ty = load_bank(cfg, f"vision_{split}"), load_bank(cfg, f"text_{split}")
    if checkpoint["encoders"] != {"vision": vx["metadata"]["encoder"], "text": ty["metadata"]["encoder"]}:
        raise ValueError("Evaluation encoders differ from the fitted training encoders")
    fwd, rev = checkpoint["forward"], checkpoint["reverse"]
    for state in (fwd, rev):
        if state["weight"].shape != (state["dim"], state["dim"]) or not torch.isfinite(state["weight"]).all():
            raise ValueError("Invalid mapping checkpoint")
    x = preprocess(vx["features"], fwd["mean_source"], fwd["dim"])
    y = preprocess(ty["features"], fwd["mean_target"], fwd["dim"])
    xr = preprocess(vx["features"], rev["mean_target"], rev["dim"])
    yr = preprocess(ty["features"], rev["mean_source"], rev["dim"])
    e = cfg["evaluation"]
    opts = {"block_size": e["batch_size"], "device": e["device"], "seed": e["shuffle_seed"]}
    result = {"setting": cfg["split"]["setting"], "split": split, "num_pairs": len(x),
              "checkpoint_sha256": file_digest(out / "mapping.pt"),
              "evaluation_banks": {n: file_digest(work_dir(cfg) / f"{n}_{split}.pt") for n in ("vision", "text")},
              "i2t": retrieval(x @ fwd["weight"], y, vx["sample_ids"], ty["sample_ids"], **opts),
              "t2i": retrieval(yr @ rev["weight"], xr, ty["sample_ids"], vx["sample_ids"], **opts),
              "note": "Separately fitted directions; saved training means; independently shuffled evaluator-only IDs."}
    write_json(out / f"retrieval_{split}.json", result)
    print(json.dumps(result, indent=2))
    return result


def audit(cfg):
    meta = audit_manifests(cfg)
    banks = {}
    for name in ("vision_train", "text_train", "vision_val", "text_val", "vision_test", "text_test"):
        if (work_dir(cfg) / f"{name}.pt").exists():
            banks[name] = list(load_bank(cfg, name)["features"].shape)
    result = {"protocol": meta, "validated_banks": banks}
    write_json(root_path(cfg["paths"]["result_dir"]) / "data_audit.json", result)
    print(json.dumps({"counts": meta["counts"], "banks": banks}, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "extract", "train", "evaluate", "audit", "all"))
    parser.add_argument("--config", default="configs/unpaired_coco2014.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--bank", choices=("vision_train", "text_train", "vision_val", "text_val", "vision_test", "text_test"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--force", action="store_true", help="Explicitly replace extracted banks or a fitted checkpoint")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    threads = int(cfg["runtime"]["torch_threads"])
    if threads < 1:
        raise ValueError("runtime.torch_threads must be positive")
    torch.set_num_threads(threads)
    with threadpool_limits(threads):
        if args.stage == "prepare":
            print(json.dumps(prepare(cfg)["counts"], indent=2))
        elif args.stage == "extract":
            if args.bank is None:
                parser.error("extract requires --bank")
            extract(cfg, args.bank, args.force)
        elif args.stage == "train":
            train(cfg, args.force)
        elif args.stage == "evaluate":
            evaluate(cfg, args.split)
        elif args.stage == "audit":
            audit(cfg)
        else:
            prepare(cfg)
            for name in ("vision_train", "text_train"):
                extract(cfg, name, args.force)
            train(cfg, args.force)
            for split in ("val", "test"):
                for modality in ("vision", "text"):
                    extract(cfg, f"{modality}_{split}", args.force)
                evaluate(cfg, split)
            audit(cfg)
