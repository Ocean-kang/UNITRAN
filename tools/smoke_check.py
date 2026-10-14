"""Portable checks for syntax and all retained experiment argument parsers."""
import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for directory in ("src", "experiments", "tests", "tools"):
        if not compileall.compile_dir(ROOT / directory, quiet=1):
            raise SystemExit(f"Compilation failed: {directory}")
    for path in sorted((ROOT / "experiments").glob("*.py")):
        subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    print("Syntax and experiment CLI checks passed")


if __name__ == "__main__":
    main()
