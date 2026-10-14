#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/smoke_check.py
python -m unittest discover -s tests -v
