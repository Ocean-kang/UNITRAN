"""Helpers for config-driven experiment entrypoints.

These helpers intentionally keep the experiment scripts thin: each script reads one YAML
file, saves the resolved config, converts selected keys to the existing argparse CLI, and
then calls the original implementation module.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except Exception:  # pragma: no cover - handled at runtime with a clearer message
    yaml = None


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


def deep_get(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def first_existing(config: Mapping[str, Any], dotted_keys: Iterable[str], default: Any = None) -> Any:
    for key in dotted_keys:
        value = deep_get(config, key, None)
        if value is not None:
            return value
    return default


def cli_args(mapping: Mapping[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in mapping.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-") if key.startswith("_") else "--" + key
        # Existing modules use underscore-style argparse flags. Keep that style stable.
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
    return parser


def infer_output_dir(config: Mapping[str, Any], *fallback_keys: str) -> str | None:
    return first_existing(config, ["experiment.output_dir", "output_dir", *fallback_keys], None)
