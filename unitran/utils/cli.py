"""CLI parsing helpers shared by lightweight tool entrypoints."""

from __future__ import annotations

from typing import Dict, List, Sequence


def parse_clusters(text: str) -> List[int]:
    text = str(text).strip()
    if not text:
        return []
    out: List[int] = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = [int(x.strip()) for x in part.split('-', 1)]
            if b < a:
                raise ValueError(f"Invalid range: {part}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def dispatch(argv: Sequence[str], command_to_module: Dict[str, str], usage: str) -> None:
    """Dispatch `argv[1]` to a module-level `main()` while preserving argparse UX."""
    import importlib
    import sys

    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        names = "\n  ".join(sorted(command_to_module))
        print(f"{usage}\n\nCommands:\n  {names}")
        return
    cmd = argv[1]
    if cmd not in command_to_module:
        names = ", ".join(sorted(command_to_module))
        raise SystemExit(f"Unknown command: {cmd}. Available commands: {names}")
    module = importlib.import_module(command_to_module[cmd])
    sys.argv = [f"{argv[0]} {cmd}", *argv[2:]]
    module.main()
