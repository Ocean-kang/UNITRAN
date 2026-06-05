#!/usr/bin/env python3
"""
Generate structured long captions and semantic atoms for a small COCO subset using local Llama3.

Input:  sample jsonl produced by build_coco_text_patch_density_eval.py --stage sample
Output: jsonl with one row per image:
  {image_id, file_name, captions, long_caption, atoms, raw_generation}

This script is intentionally text-only: it uses the original COCO captions as input and does not use
vision cluster/top-k patch information. Use it to test whether text expansion alone improves the
text distribution's match to DINOv2 patch-token distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm.auto import tqdm


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def existing_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add(int(row["image_id"]))
            except Exception:
                continue
    return done


def build_prompt(captions: List[str], max_atoms: int, num_atoms: int = 0) -> str:
    caps = "\n".join([f"{i+1}. {c}" for i, c in enumerate(captions)])
    atom_rule = f"exactly {num_atoms}" if num_atoms > 0 else f"8 to {max_atoms}"
    return f"""You are rewriting MSCOCO image captions for a vision-language representation experiment.

Original captions:
{caps}

Task:
1. Write one grounded long caption in English. Only use details that are directly supported by the captions or highly conservative visual implications.
2. Split the result into short semantic atoms. Each atom should describe ONE visual unit: object, part, attribute, material, texture, action, spatial relation, or background/stuff.
3. Avoid hallucinated details such as exact colors, brands, counts, or locations unless the captions clearly support them.
4. Output valid JSON only. Do not add markdown.

JSON schema:
{{
  "long_caption": "...",
  "atoms": ["short atom 1", "short atom 2", "..."]
}}

Constraints:
- Number of atoms: {atom_rule}.
- Each atom: 4 to 18 words.
- No duplicate atoms.
"""


def parse_json_generation(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Prefer fenced/clean JSON, but tolerate extra prefix/suffix.
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in generation")
    obj = json.loads(match.group(0))
    long_caption = str(obj.get("long_caption", "")).strip()
    atoms = obj.get("atoms", [])
    if not isinstance(atoms, list):
        atoms = []
    atoms = [str(a).strip() for a in atoms if str(a).strip()]
    # Deduplicate while preserving order.
    seen = set()
    atoms2: List[str] = []
    for a in atoms:
        key = a.lower()
        if key not in seen:
            atoms2.append(a)
            seen.add(key)
    return {"long_caption": long_caption, "atoms": atoms2}


def fallback_atoms(captions: List[str]) -> List[str]:
    atoms: List[str] = []
    for c in captions:
        c = c.strip()
        if c and c.lower() not in {a.lower() for a in atoms}:
            atoms.append(c)
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate COCO long captions and semantic atoms using local Llama3.")
    parser.add_argument("--sample_jsonl", required=True, type=str)
    parser.add_argument("--out_jsonl", required=True, type=str)
    parser.add_argument("--llama_root", required=True, type=str, help="Path containing the llama package, e.g. ./llama3")
    parser.add_argument("--ckpt_dir", required=True, type=str)
    parser.add_argument("--tokenizer_path", required=True, type=str)
    parser.add_argument("--max_seq_len", default=2048, type=int)
    parser.add_argument("--max_batch_size", default=4, type=int)
    parser.add_argument("--max_gen_len", default=512, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--max_atoms", default=18, type=int)
    parser.add_argument("--num_atoms", default=0, type=int, help="If >0, ask Llama3 to generate exactly this many atoms per image.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.llama_root).resolve()))
    llama_parent = Path(__file__).resolve().parents[2] / "third_party"
    sys.path.insert(0, str(llama_parent))
    from third_party.llama import Llama  # type: ignore

    rows = load_jsonl(Path(args.sample_jsonl))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out_jsonl)
    done = existing_ids(out_path) if args.resume else set()
    if out_path.exists() and not args.resume:
        raise FileExistsError(f"{out_path} exists. Use --resume or remove it.")

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        model_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    )

    batch: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="llama3 recaption"):
        if int(row["image_id"]) in done:
            continue
        batch.append(row)
        if len(batch) < args.max_batch_size:
            continue

        dialogs = [
            [
                {"role": "system", "content": "You produce grounded JSON for image-caption rewriting."},
                {"role": "user", "content": build_prompt(r["captions"], args.max_atoms, args.num_atoms)},
            ]
            for r in batch
        ]
        outputs = generator.chat_completion(dialogs, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p)
        out_rows = []
        for r, o in zip(batch, outputs):
            raw = o["generation"]["content"]
            try:
                parsed = parse_json_generation(raw)
            except Exception:
                parsed = {"long_caption": " ".join(r["captions"]), "atoms": fallback_atoms(r["captions"])}
            out_rows.append({**r, **parsed, "raw_generation": raw})
        append_jsonl(out_path, out_rows)
        batch = []

    if batch:
        dialogs = [
            [
                {"role": "system", "content": "You produce grounded JSON for image-caption rewriting."},
                {"role": "user", "content": build_prompt(r["captions"], args.max_atoms, args.num_atoms)},
            ]
            for r in batch
        ]
        outputs = generator.chat_completion(dialogs, max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p)
        out_rows = []
        for r, o in zip(batch, outputs):
            raw = o["generation"]["content"]
            try:
                parsed = parse_json_generation(raw)
            except Exception:
                parsed = {"long_caption": " ".join(r["captions"]), "atoms": fallback_atoms(r["captions"])}
            out_rows.append({**r, **parsed, "raw_generation": raw})
        append_jsonl(out_path, out_rows)

    print(f"[save] {out_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
