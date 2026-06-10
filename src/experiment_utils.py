"""Helpers for config-driven experiment entrypoints.

These helpers intentionally keep the experiment scripts thin: each script reads one YAML
file, resolves optional dataset presets and command-line overrides, saves the resolved
config, converts selected keys to the existing argparse CLI, and then calls the original
implementation module.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

try:
    import yaml
except Exception:  # pragma: no cover - handled at runtime with a clearer message
    yaml = None


_SECTION_KEYS = {
    "data",
    "vision",
    "text",
    "feature",
    "experiment",
    "cluster",
    "evaluation",
    "sample",
    "prototype",
    "runtime",
    "llm",
    "mapping",
    "training",
    "visualization",
    "embedding",
}

_DATA_KEYS = {
    "coco_root",
    "split",
    "ann_path",
    "image_dir",
    "mask_dir",
    "caption_json",
    "annotation_type",
    "image_root",
    "num_images",
    "max_images",
    "limit",
}


_FLAT_DATASET_KEY_MAP = {
    # data
    "coco_root": ("data.coco_root",),
    "split": ("data.split",),
    "ann_path": ("data.ann_path",),
    "image_dir": ("data.image_dir",),
    "mask_dir": ("data.mask_dir",),
    "caption_json": ("data.caption_json",),
    "annotation_type": ("data.annotation_type",),
    "image_root": ("data.image_root",),
    "num_images": ("data.num_images",),
    "max_images": ("data.max_images",),
    "limit": ("data.limit",),
    # vision / feature paths
    "feature_dir": ("vision.feature_dir",),
    "input_dir": ("vision.feature_dir",),
    "vision_feature_dir": ("vision.feature_dir", "vision.local_feature_dir"),
    "raw_feature_dir": ("vision.raw_feature_dir",),
    "local_feature_dir": ("vision.local_feature_dir",),
    "local_input_dir": ("vision.local_feature_dir",),
    "feature_out_dir": ("feature.out_dir",),
    # text paths
    "text_feature_dir": ("text.output_dir",),
    "recaption_jsonl": ("text.recaption_jsonl",),
    "sample_jsonl": ("sample.out_jsonl", "evaluation.sample_jsonl", "text.sample_jsonl"),
    "text_atoms_jsonl": ("text.out_jsonl",),
    # generic output directory used by wrappers to decide where resolved_config/report/checkpoints go.
    # Reusable feature outputs should use feature_out_dir when they differ from report outputs.
    "output_dir": ("experiment.output_dir",),
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Install with: pip install pyyaml")
        obj = yaml.safe_load(text)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"Top-level config must be a mapping, got {type(obj)!r}")
    return obj


def save_resolved_config(config: Mapping[str, Any], output_dir: str | Path | None) -> None:
    if not output_dir:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resolved_config.yaml"
    if yaml is not None:
        path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_scalar(value: str) -> Any:
    """Parse a --set scalar into bool, int, float, None, or str.

    This intentionally stays small and predictable instead of implementing a full schema.
    """

    raw = value.strip()
    lowered = raw.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", "~"}:
        return None
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    if re.fullmatch(r"[-+]?\d+", raw):
        try:
            return int(raw)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", raw):
        try:
            return float(raw)
        except ValueError:
            pass
    return raw


def deep_get(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def get_by_dotted_key(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Alias used by config/path call sites that prefer explicit dotted-key wording."""

    return deep_get(config, dotted_key, default)


def deep_set(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> MutableMapping[str, Any]:
    """Set a nested config value by dotted key, e.g. data.num_images=5000."""

    if not dotted_key or dotted_key.startswith(".") or dotted_key.endswith("."):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    cur: MutableMapping[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None:
            nxt = {}
            cur[part] = nxt
        if not isinstance(nxt, MutableMapping):
            raise TypeError(f"Cannot set {dotted_key!r}: {part!r} is not a mapping")
        cur = nxt
    cur[parts[-1]] = value
    return config


def _deep_merge(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), MutableMapping):
            _deep_merge(dst[key], value)  # type: ignore[index]
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply --set KEY=VALUE overrides to a config copy."""

    resolved = copy.deepcopy(dict(config))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set override {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        deep_set(resolved, key.strip(), parse_scalar(value))
    return resolved


def _apply_flat_dataset_key(config: MutableMapping[str, Any], key: str, value: Any) -> bool:
    dotted_keys = _FLAT_DATASET_KEY_MAP.get(key)
    if not dotted_keys:
        return False
    for dotted_key in dotted_keys:
        deep_set(config, dotted_key, value)
    return True


def apply_active_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge config['datasets'][config['active_dataset']] into a config copy.

    The function supports both simple flat dataset presets and explicit nested presets.
    Nested section keys such as ``data:``, ``vision:``, ``text:``, ``feature:`` and
    ``experiment:`` are deep-merged directly. Common flat keys such as ``coco_root``,
    ``feature_dir``, ``local_feature_dir`` and ``output_dir`` are routed to the sections
    already used by the experiment wrappers.
    """

    resolved = copy.deepcopy(dict(config))
    active = resolved.get("active_dataset")
    if active in (None, ""):
        return resolved
    datasets = resolved.get("datasets", {})
    if not isinstance(datasets, Mapping):
        raise TypeError("config['datasets'] must be a mapping when active_dataset is set")
    if active not in datasets:
        raise KeyError(f"active_dataset={active!r} not found in config['datasets']")
    preset = datasets[active]
    if not isinstance(preset, Mapping):
        raise TypeError(f"datasets[{active!r}] must be a mapping")

    for key, value in preset.items():
        if key in _SECTION_KEYS and isinstance(value, Mapping):
            section = resolved.setdefault(key, {})
            if not isinstance(section, MutableMapping):
                raise TypeError(f"Cannot merge dataset section {key!r}: config value is not a mapping")
            _deep_merge(section, value)
        elif not _apply_flat_dataset_key(resolved, key, value):
            # Unknown flat keys are kept under data by default. This preserves lightweight
            # dataset metadata without requiring a schema system.
            if key in _DATA_KEYS:
                deep_set(resolved, f"data.{key}", value)
            else:
                deep_set(resolved, f"data.{key}", value)
    return resolved


def resolve_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """Load config, apply active_dataset first, then all other --set overrides.

    ``--set active_dataset=mscoco17`` must be visible before dataset merging, while
    overrides such as ``--set feature.out_dir=...`` should win after the preset merge.
    """

    overrides = list(overrides or [])
    active_overrides = [x for x in overrides if x.split("=", 1)[0].strip() == "active_dataset"]
    other_overrides = [x for x in overrides if x.split("=", 1)[0].strip() != "active_dataset"]
    cfg = load_config(path)
    cfg = apply_overrides(cfg, active_overrides)
    cfg = apply_active_dataset(cfg)
    cfg = apply_overrides(cfg, other_overrides)
    return cfg


def first_existing(config: Mapping[str, Any], dotted_keys: Iterable[str], default: Any = None) -> Any:
    for key in dotted_keys:
        value = deep_get(config, key, None)
        if value is not None:
            return value
    return default


def get_feature_out_dir(config: Mapping[str, Any], default: Any = None) -> Any:
    return first_existing(config, ["feature.out_dir", "feature_dir", "out_dir"], default)


def get_output_dir(config: Mapping[str, Any], default: Any = None) -> Any:
    return first_existing(config, ["experiment.output_dir", "output_dir", "out_dir"], default)


def cli_args(mapping: Mapping[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in mapping.items():
        if value is None or value is False:
            continue
        flag = "--" + key
        if value is True:
            args.append(flag)
        elif isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(x) for x in value)
        else:
            args.extend([flag, str(value)])
    return args


def call_main(module_name: str, args: list[str]) -> None:
    module = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    sys.argv = [module_name, *args]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def add_config_arg(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON experiment config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values, e.g. --set data.num_images=5000",
    )
    return parser


def infer_output_dir(config: Mapping[str, Any], *fallback_keys: str) -> str | None:
    return first_existing(config, ["experiment.output_dir", "output_dir", *fallback_keys], None)
