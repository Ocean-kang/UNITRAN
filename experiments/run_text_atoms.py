#!/usr/bin/env python3
"""Config-driven LLM recaption / semantic atom generation."""

from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_utils import add_config_arg, call_main, cli_args, first_existing, infer_output_dir, load_config, save_resolved_config


def main() -> None:
    parser = add_config_arg("Generate structured long captions and semantic atoms.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = infer_output_dir(cfg, "text.output_dir")
    save_resolved_config(cfg, out_dir)

    mapping = {
        "sample_jsonl": first_existing(cfg, ["text.sample_jsonl", "sample_jsonl"], None),
        "out_jsonl": first_existing(cfg, ["text.out_jsonl", "out_jsonl"], None),
        "llama_root": first_existing(cfg, ["llm.llama_root", "llama_root"], "external/llama"),
        "ckpt_dir": first_existing(cfg, ["llm.ckpt_dir", "ckpt_dir"], None),
        "tokenizer_path": first_existing(cfg, ["llm.tokenizer_path", "tokenizer_path"], None),
        "max_seq_len": first_existing(cfg, ["llm.max_seq_len", "max_seq_len"], 2048),
        "max_batch_size": first_existing(cfg, ["llm.max_batch_size", "max_batch_size"], 4),
        "max_gen_len": first_existing(cfg, ["llm.max_gen_len", "max_gen_len"], 512),
        "temperature": first_existing(cfg, ["llm.temperature", "temperature"], 0.2),
        "top_p": first_existing(cfg, ["llm.top_p", "top_p"], 0.9),
        "max_atoms": first_existing(cfg, ["text.max_atoms", "max_atoms"], 18),
        "num_atoms": first_existing(cfg, ["text.num_atoms", "num_atoms"], 0),
        "resume": first_existing(cfg, ["runtime.resume", "resume"], False),
        "limit": first_existing(cfg, ["data.limit", "limit"], 0),
    }
    missing = [k for k in ("sample_jsonl", "out_jsonl", "ckpt_dir", "tokenizer_path") if mapping[k] in (None, "")]
    if missing:
        raise SystemExit(f"Missing required config keys for text atoms: {', '.join(missing)}")
    call_main("src.features.text_atoms", cli_args(mapping))


if __name__ == "__main__":
    main()
