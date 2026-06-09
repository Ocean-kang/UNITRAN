#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python experiments/run_text_atoms.py --config configs/text_atoms_llama.yaml
