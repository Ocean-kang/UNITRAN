#!/usr/bin/env python3
"""Single entry point for manifest preparation, encoders, mapping, and evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unpaired.pipeline import main

if __name__ == "__main__":
    main()
