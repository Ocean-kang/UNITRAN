#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m compileall -q src experiments
python experiments/extract_dinov2_features.py --help >/dev/null
python experiments/run_global_patch_cluster.py --help >/dev/null
python experiments/run_local_patch_cluster.py --help >/dev/null
python experiments/run_text_atoms.py --help >/dev/null
python experiments/run_density_eval.py --help >/dev/null
python experiments/run_visualize_cluster.py --help >/dev/null
python experiments/extract_eval_prototypes.py --help >/dev/null
python experiments/train_mapping.py --help >/dev/null
echo "smoke check passed"
