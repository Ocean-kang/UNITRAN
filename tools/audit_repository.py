"""Inventory every tracked file and Python call site at a fixed Git ref, without importing it."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def inventory(ref):
    paths = git("ls-tree", "-r", "--name-only", ref).decode().splitlines()
    files = []
    for path in paths:
        raw = git("show", f"{ref}:{path}")
        text = raw.decode("utf-8")
        item = {"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                "lines": len(text.splitlines()), "bytes": len(raw)}
        if path.endswith(".py"):
            tree = ast.parse(text, filename=path)
            item["imports"] = sorted({n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                                     {alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names})
            item["definitions"] = [{"name": n.name, "line": n.lineno}
                                    for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            item["calls"] = [{"callee": ast.unparse(n.func), "line": n.lineno}
                              for n in ast.walk(tree) if isinstance(n, ast.Call)]
            item["dynamic_modules"] = sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                                              and isinstance(n.value, str) and n.value.startswith("src.")})
            item["standalone_cli"] = '__name__ == "__main__"' in text
        files.append(item)
    return {"commit": git("rev-parse", ref).decode().strip(), "files": files,
            "history": git("log", "--format=%h %s", ref).decode().splitlines()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="6160deceb3ca457d4ac7b6e2e31ece497f1b38ad")
    parser.add_argument("--output", default="outputs/audit_repository.json")
    parser.add_argument("--summary", default=None, help="Optional tracked Markdown inventory")
    args = parser.parse_args()
    result = inventory(args.ref)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.summary:
        lines = [f"# Baseline file inventory\n\nAudited commit: `{result['commit']}`.\n",
                 "All tracked files are read; Python ASTs include every definition/call site in the reproducible JSON report.\n",
                 "| File | Lines | Internal imports / dispatch | Disposition |\n|---|---:|---|---|\n"]
        for item in result["files"]:
            dependencies = [name for name in item.get("imports", []) if name.startswith("src.") or name in {"common", "metrics", "llama"}]
            dependencies += item.get("dynamic_modules", [])
            if item["path"].startswith("experiments/unpaired_mpnet/"):
                disposition = "Merged into core; removed"
            elif item["path"] in {"src/features/dinov2.py", "src/experiment_utils.py"}:
                disposition = "Shared and retained"
            elif item["path"] == "src/training/unitran_mapping.py":
                disposition = "Legacy adapter; duplicated mini core removed"
            elif item["path"].startswith(("src/", "configs/", "experiments/", "scripts/", "external/")):
                disposition = "Retained optional workflow / CLI"
            else:
                disposition = "Retained or updated documentation/environment"
            lines.append(f"| `{item['path']}` | {item['lines']} | {', '.join(sorted(set(dependencies))) or '—'} | {disposition} |\n")
        (ROOT / args.summary).write_text("".join(lines), encoding="utf-8")
    print(f"Inventoried {len(result['files'])} files at {result['commit']}")


if __name__ == "__main__":
    main()
