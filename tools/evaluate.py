#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python tools/evaluate.py <command> [args]\n\nCommands:\n  distribution\n  text-density sample|features|report")
        return
    cmd = sys.argv[1]
    if cmd == "distribution":
        from unitran.evaluation.distribution import main as m
        sys.argv = [sys.argv[0] + " distribution", *sys.argv[2:]]
        m()
    elif cmd == "text-density":
        from unitran.evaluation.text_density import main as m
        rest = sys.argv[2:]
        if rest and rest[0] in {"sample", "features", "report"}:
            stage = rest[0]
            rest = ["--stage", stage, *rest[1:]]
        sys.argv = [sys.argv[0] + " text-density", *rest]
        m()
    else:
        raise SystemExit(f"Unknown command: {cmd}. Available commands: distribution, text-density")

if __name__ == "__main__":
    main()
