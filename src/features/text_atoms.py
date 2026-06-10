#!/usr/bin/env python3
"""
Generate structured long captions and semantic atoms for a small COCO subset using local Llama3.

Input:  sample jsonl produced by local density sampling.
Output: jsonl with one row per image:
  {image_id, file_name, captions, long_caption, atoms, raw_generation}

The script supports two Llama backends:
  - hf:   HuggingFace Transformers format, e.g. Meta-Llama-3-8B-Instruct-HF
  - meta: Meta official llama repository/checkpoint format

For new experiments, prefer backend=hf because it does not require the external Meta llama code package.
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


# def build_prompt(captions: List[str], max_atoms: int, num_atoms: int = 0) -> str:
#     caps = "\n".join([f"{i+1}. {c}" for i, c in enumerate(captions)])
#     atom_rule = f"exactly {num_atoms}" if num_atoms > 0 else f"8 to {max_atoms}"
#     return f"""You are rewriting MSCOCO image captions for a vision-language representation experiment.

# Original captions:
# {caps}

# Task:
# 1. Write one grounded long caption in English. Only use details that are directly supported by the captions or highly conservative visual implications.
# 2. Split the result into short semantic atoms. Each atom should describe ONE visual unit: object, part, attribute, material, texture, action, spatial relation, or background/stuff.
# 3. Avoid hallucinated details such as exact colors, brands, counts, or locations unless the captions clearly support them.
# 4. Output valid JSON only. Do not add markdown.

# JSON schema:
# {{
#   "long_caption": "...",
#   "atoms": ["short atom 1", "short atom 2", "..."]
# }}

# Constraints: 
# - Number of atoms: {atom_rule}.
# - Each atom: 4 to 18 words.
# - No duplicate atoms.
# """

def build_prompt(captions: List[str], max_atoms: int, num_atoms: int = 0) -> str:
    caps = "\n".join([f"{i+1}. {c}" for i, c in enumerate(captions)])
    atom_rule = f"exactly {num_atoms}" if num_atoms > 0 else f"8 to {max_atoms}"

    return f"""You construct dense visual semantic atoms for matching DINOv2 patch/region features.

Original MSCOCO captions:
{caps}

Goal:
Generate short, grounded, visually observable text atoms that can correspond to local image patches, object parts, textures, materials, shapes, surfaces, actions, spatial relations, or background/stuff regions.

Important rules:
1. Use only information supported by the captions or very conservative visual implications.
2. Prefer local component-level phrases over full-sentence captions.
3. Each atom should describe ONE visual unit.
4. Generate positive visual phrases only.
5. Do not use negation, contrast, uncertainty, or non-visual reasoning.
6. Do not generate encyclopedic definitions or object functions.
7. Avoid hallucinated exact colors, brands, numbers, locations, or identities unless clearly supported.
8. Include both foreground objects and background/stuff regions when supported.

Good atom examples:
- human hand region
- animal fur texture
- round vehicle wheel
- wooden table surface
- grassy ground area
- person standing pose
- object boundary edge
- background vegetation texture

Bad atom examples:
- not a dog
- probably in Europe
- used for transportation
- unlike other animals
- beautiful outdoor scene
- the image may contain something

Output valid JSON only. Do not add markdown.

JSON schema:
{{
  "long_caption": "one conservative grounded caption summarizing the image",
  "atoms": ["short visual atom 1", "short visual atom 2", "..."]
}}

Constraints:
- Number of atoms: {atom_rule}.
- Each atom: 2 to 8 words.
- No duplicate atoms.
- Atoms should be concrete visual phrases, not full sentences.
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


def _chat_dialog(row: Dict[str, Any], max_atoms: int, num_atoms: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You produce grounded JSON for image-caption rewriting."},
        {"role": "user", "content": build_prompt(row["captions"], max_atoms, num_atoms)},
    ]


def _torch_dtype(name: str):
    import torch

    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"auto", "none"}:
        return "auto"
    raise ValueError(f"Unsupported torch dtype: {name}")


def build_hf_generator(args: argparse.Namespace):
    """Load a HuggingFace-format Llama model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(args.ckpt_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"HF model directory not found: {model_dir}")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"{model_dir} does not look like a HuggingFace model directory. "
            "Expected config.json. If this is Meta original format, use --backend meta."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "dtype": _torch_dtype(args.torch_dtype),
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    if not args.device_map and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return tokenizer, model


def hf_chat_completion(tokenizer: Any, model: Any, batch: List[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    import torch

    dialogs = [_chat_dialog(r, args.max_atoms, args.num_atoms) for r in batch]
    prompts = [
        tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
        for dialog in dialogs
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_seq_len,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids:
        eos_ids.append(eot_id)
    eos_arg = eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None)

    # HF Llama 的 generation_config.json 里可能自带 max_length=4096。
    # 我们只用 max_new_tokens 控制生成长度，所以这里清掉 max_length，避免 warning。
    model.generation_config.max_length = None

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_gen_len,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            eos_token_id=eos_arg,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, prompt_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,)


def build_meta_generator(args: argparse.Namespace):
    """Load the original Meta llama backend kept under external/llama or third_party/llama."""
    llama_root = Path(args.llama_root).resolve() if args.llama_root else None
    if llama_root is not None:
        # If llama_root is external/llama, its parent must be importable for `from llama import Llama`.
        sys.path.insert(0, str(llama_root.parent))
        sys.path.insert(0, str(llama_root))

    # Support both current external/llama layout and older third_party.llama layout.
    try:
        from llama import Llama  # type: ignore
    except Exception:
        llama_parent = Path(__file__).resolve().parents[2] / "third_party"
        sys.path.insert(0, str(llama_parent))
        from third_party.llama import Llama  # type: ignore

    if not args.tokenizer_path:
        raise ValueError("--tokenizer_path is required for --backend meta")

    return Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        model_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    )


def meta_chat_completion(generator: Any, batch: List[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    dialogs = [_chat_dialog(r, args.max_atoms, args.num_atoms) for r in batch]
    outputs = generator.chat_completion(
        dialogs,
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return [o["generation"]["content"] for o in outputs]


def parse_backend(backend: str, ckpt_dir: str) -> str:
    if backend != "auto":
        return backend
    ckpt = Path(ckpt_dir)
    if (ckpt / "config.json").exists():
        return "hf"
    return "meta"


def process_batch(batch: List[Dict[str, Any]], raw_texts: List[str]) -> List[Dict[str, Any]]:
    out_rows = []
    for r, raw in zip(batch, raw_texts):
        try:
            parsed = parse_json_generation(raw)
        except Exception:
            parsed = {"long_caption": " ".join(r["captions"]), "atoms": fallback_atoms(r["captions"])}
        out_rows.append({**r, **parsed, "raw_generation": raw})
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate COCO long captions and semantic atoms using local Llama3.")
    parser.add_argument("--sample_jsonl", required=True, type=str)
    parser.add_argument("--out_jsonl", required=True, type=str)
    parser.add_argument("--backend", default="auto", choices=["auto", "hf", "meta"], help="Llama backend. Use hf for HuggingFace-format weights.")
    parser.add_argument("--llama_root", default="external/llama", type=str, help="Meta backend only: path containing the llama package.")
    parser.add_argument("--ckpt_dir", required=True, type=str, help="HF model dir or Meta original checkpoint dir.")
    parser.add_argument("--tokenizer_path", default=None, type=str, help="Meta backend only: tokenizer.model path.")
    parser.add_argument("--max_seq_len", default=2048, type=int)
    parser.add_argument("--max_batch_size", default=4, type=int)
    parser.add_argument("--max_gen_len", default=512, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--max_atoms", default=18, type=int)
    parser.add_argument("--num_atoms", default=0, type=int, help="If >0, ask Llama3 to generate exactly this many atoms per image.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--torch_dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], help="HF backend only.")
    parser.add_argument("--device_map", default="auto", type=str, help="HF backend only. Use auto, cuda:0, or empty string.")
    parser.add_argument("--local_files_only", action="store_true", help="HF backend only: do not try to download files.")
    parser.add_argument("--trust_remote_code", action="store_true", help="HF backend only: allow custom model code if needed.")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.sample_jsonl))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out_jsonl)
    done = existing_ids(out_path) if args.resume else set()
    if out_path.exists() and not args.resume:
        raise FileExistsError(f"{out_path} exists. Use --resume or remove it.")

    backend = parse_backend(args.backend, args.ckpt_dir)
    print(f"[llama] backend={backend} ckpt_dir={args.ckpt_dir}")
    if backend == "hf":
        tokenizer, model = build_hf_generator(args)
        generate_raw = lambda batch: hf_chat_completion(tokenizer, model, batch, args)
    else:
        generator = build_meta_generator(args)
        generate_raw = lambda batch: meta_chat_completion(generator, batch, args)

    batch: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="llama3 recaption"):
        if int(row["image_id"]) in done:
            continue
        batch.append(row)
        if len(batch) < args.max_batch_size:
            continue
        append_jsonl(out_path, process_batch(batch, generate_raw(batch)))
        batch = []

    if batch:
        append_jsonl(out_path, process_batch(batch, generate_raw(batch)))

    print(f"[save] {out_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
