"""Configuration with small, explicit presets and repository-relative paths."""
from pathlib import Path

from src.experiment_utils import apply_overrides, load_config as read_config

ROOT = Path(__file__).resolve().parents[2]

FIELDS = {
    "paths": {"coco_root", "work_dir", "result_dir", "spc_paragraphs", "spc_image_metadata"},
    "split": {"setting", "seed", "expected_train_images", "expected_eval_images", "train_size_per_side", "val_size", "test_size"},
    "vision": {"model", "image_size", "pooling", "repo", "weights", "revision", "batch_size", "num_workers"},
    "text": {"model", "revision", "image_batch_size", "text_batch_size"},
    "runtime": {"device", "torch_threads"},
    "mapping": {"seed", "num_anchor_runs", "num_clusters", "qap_runs", "anchor_subsample", "initial_k", "refine_iters",
                "refine_sample", "refine_k", "refine2_clusters", "alpha", "similarity_batch_size", "similarity_device"},
    "evaluation": {"batch_size", "device", "shuffle_seed"},
}


def root_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path, overrides=(), _seen=()):
    path = root_path(path).resolve()
    if path in _seen:
        raise ValueError("Cyclic config inheritance")
    cfg = read_config(path)
    parent = cfg.pop("_base_", None)
    if parent:
        cfg = merge(load_config(path.parent / parent, _seen=(*_seen, path)), cfg)
    cfg = apply_overrides(cfg, overrides)
    if set(cfg) != set(FIELDS):
        raise ValueError("Missing or unknown config section")
    for section, keys in FIELDS.items():
        if not isinstance(cfg[section], dict) or set(cfg[section]) - keys:
            raise ValueError(f"Invalid/unknown fields in {section}")
    if cfg["split"]["setting"] not in {"disjoint", "cross_dataset"}:
        raise ValueError("setting must be disjoint or cross_dataset")
    if cfg["split"]["setting"] == "cross_dataset":
        for key in ("spc_paragraphs", "spc_image_metadata"):
            if not cfg["paths"].get(key):
                raise ValueError(f"Cross-dataset setting requires paths.{key}")
    for key in ("train_size_per_side", "val_size", "test_size"):
        size = cfg["split"].get(key)
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 2):
            raise ValueError(f"split.{key} must be null or an integer >= 2")
    if cfg["vision"]["pooling"] not in {"patch_mean", "all_token_mean", "cls"}:
        raise ValueError("Unknown vision pooling")
    if int(cfg["vision"]["image_size"]) % 14 or int(cfg["vision"]["image_size"]) < 14:
        raise ValueError("DINOv2 image_size must be a positive multiple of 14")
    for section, keys in (("vision", ("batch_size",)), ("text", ("image_batch_size", "text_batch_size")),
                          ("runtime", ("torch_threads",)), ("evaluation", ("batch_size",))):
        for key in keys:
            if not isinstance(cfg[section][key], int) or cfg[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if not isinstance(cfg["vision"]["num_workers"], int) or cfg["vision"]["num_workers"] < 0:
        raise ValueError("vision.num_workers must be a nonnegative integer")
    return cfg
