# Baseline file inventory

Audited commit: `6160deceb3ca457d4ac7b6e2e31ece497f1b38ad`.
All tracked files are read; Python ASTs include every definition/call site in the reproducible JSON report.
| File | Lines | Internal imports / dispatch | Disposition |
|---|---:|---|---|
| `.gitignore` | 225 | — | Retained or updated documentation/environment |
| `LICENSE` | 21 | — | Retained or updated documentation/environment |
| `README.md` | 301 | — | Retained or updated documentation/environment |
| `conda.yaml` | 23 | — | Retained or updated documentation/environment |
| `configs/cluster_global.yaml` | 32 | — | Retained optional workflow / CLI |
| `configs/cluster_local.yaml` | 29 | — | Retained optional workflow / CLI |
| `configs/density_local.yaml` | 48 | — | Retained optional workflow / CLI |
| `configs/eval_prototypes.yaml` | 53 | — | Retained optional workflow / CLI |
| `configs/extract_clip_dinov2_patch_bank.yaml` | 43 | — | Retained optional workflow / CLI |
| `configs/extract_dinov2.yaml` | 43 | — | Retained optional workflow / CLI |
| `configs/text_atoms.yaml` | 41 | — | Retained optional workflow / CLI |
| `configs/train_mapping.yaml` | 60 | — | Retained optional workflow / CLI |
| `configs/train_mapping_clip_dinov2.yaml` | 58 | — | Retained optional workflow / CLI |
| `configs/train_mapping_eval.yaml` | 58 | — | Retained optional workflow / CLI |
| `configs/visualize_cluster.yaml` | 33 | — | Retained optional workflow / CLI |
| `docs/experiment_pipeline.md` | 25 | — | Retained or updated documentation/environment |
| `docs/known_issues.md` | 17 | — | Retained or updated documentation/environment |
| `docs/result_format.md` | 123 | — | Retained or updated documentation/environment |
| `experiments/extract_clip_dinov2_patch_bank.py` | 49 | src.experiment_utils, src.features.clip_dinov2_patch_bank | Retained optional workflow / CLI |
| `experiments/extract_dinov2_features.py` | 53 | src.experiment_utils, src.features.dinov2 | Retained optional workflow / CLI |
| `experiments/extract_eval_prototypes.py` | 50 | src.evaluation.eval_prototypes, src.experiment_utils | Retained optional workflow / CLI |
| `experiments/run_density_eval.py` | 133 | src.evaluation.local_density, src.evaluation.text_density, src.experiment_utils | Retained optional workflow / CLI |
| `experiments/run_global_patch_cluster.py` | 45 | src.clustering.stream_kmeans, src.experiment_utils | Retained optional workflow / CLI |
| `experiments/run_local_patch_cluster.py` | 40 | src.clustering.local_kmeans, src.experiment_utils | Retained optional workflow / CLI |
| `experiments/run_text_atoms.py` | 51 | src.experiment_utils, src.features.text_atoms | Retained optional workflow / CLI |
| `experiments/run_visualize_cluster.py` | 114 | src.experiment_utils, src.visualization.cluster_center_patches, src.visualization.cluster_patches, src.visualization.embedding_2d, src.visualization.image_cls_attention, src.visualization.image_local_centers_attention | Retained optional workflow / CLI |
| `experiments/train_mapping.py` | 67 | src.experiment_utils, src.training.unitran_mapping | Retained optional workflow / CLI |
| `experiments/unpaired_mpnet/README.md` | 162 | — | Merged into core; removed |
| `experiments/unpaired_mpnet/common.py` | 121 | — | Merged into core; removed |
| `experiments/unpaired_mpnet/config.yaml` | 52 | — | Merged into core; removed |
| `experiments/unpaired_mpnet/evaluate.py` | 53 | common, metrics | Merged into core; removed |
| `experiments/unpaired_mpnet/extract_text.py` | 55 | common | Merged into core; removed |
| `experiments/unpaired_mpnet/extract_vision.py` | 65 | common, src.features.dinov2 | Merged into core; removed |
| `experiments/unpaired_mpnet/metrics.py` | 63 | — | Merged into core; removed |
| `experiments/unpaired_mpnet/run.sh` | 11 | — | Merged into core; removed |
| `experiments/unpaired_mpnet/split_coco.py` | 101 | common | Merged into core; removed |
| `experiments/unpaired_mpnet/test_small.py` | 47 | metrics | Merged into core; removed |
| `experiments/unpaired_mpnet/train.py` | 123 | common, metrics, src.training.unitran_mapping | Merged into core; removed |
| `experiments/visualize_disentangled_heads.py` | 649 | — | Retained optional workflow / CLI |
| `external/llama/__init__.py` | 6 | — | Retained optional workflow / CLI |
| `external/llama/generation_feat.py` | 955 | — | Retained optional workflow / CLI |
| `external/llama/model.py` | 350 | — | Retained optional workflow / CLI |
| `external/llama/tokenizer.py` | 229 | — | Retained optional workflow / CLI |
| `scripts/run_1_extract.sh` | 4 | — | Retained optional workflow / CLI |
| `scripts/run_2_global_cluster.sh` | 4 | — | Retained optional workflow / CLI |
| `scripts/run_2_local_cluster.sh` | 4 | — | Retained optional workflow / CLI |
| `scripts/run_3_text_atoms.sh` | 4 | — | Retained optional workflow / CLI |
| `scripts/run_4_density_eval.sh` | 4 | — | Retained optional workflow / CLI |
| `scripts/run_5_visualize.sh` | 4 | — | Retained optional workflow / CLI |
| `scripts/run_clip_dinov2_patch_bank.sh` | 4 | — | Retained optional workflow / CLI |
| `scripts/run_clip_dinov2_train_mapping.sh` | 4 | — | Retained optional workflow / CLI |
| `scripts/smoke_check.sh` | 13 | — | Retained optional workflow / CLI |
| `src/__init__.py` | 1 | — | Retained optional workflow / CLI |
| `src/clustering/__init__.py` | 1 | — | Retained optional workflow / CLI |
| `src/clustering/local_kmeans.py` | 524 | — | Retained optional workflow / CLI |
| `src/clustering/stream_kmeans.py` | 560 | — | Retained optional workflow / CLI |
| `src/datasets/__init__.py` | 0 | — | Retained optional workflow / CLI |
| `src/datasets/coco.py` | 92 | — | Retained optional workflow / CLI |
| `src/evaluation/__init__.py` | 0 | — | Retained optional workflow / CLI |
| `src/evaluation/eval_prototypes.py` | 425 | src.evaluation.text_density, src.features.dinov2 | Retained optional workflow / CLI |
| `src/evaluation/local_density.py` | 326 | src.evaluation.text_density | Retained optional workflow / CLI |
| `src/evaluation/text_density.py` | 498 | — | Retained optional workflow / CLI |
| `src/experiment_utils.py` | 325 | — | Shared and retained |
| `src/features/__init__.py` | 0 | — | Retained optional workflow / CLI |
| `src/features/clip_dinov2_patch_bank.py` | 320 | src.features.dinov2 | Retained optional workflow / CLI |
| `src/features/dinov2.py` | 849 | — | Shared and retained |
| `src/features/patch_store.py` | 296 | — | Retained optional workflow / CLI |
| `src/features/text_atoms.py` | 401 | llama | Retained optional workflow / CLI |
| `src/training/__init__.py` | 0 | — | Retained optional workflow / CLI |
| `src/training/unitran_mapping.py` | 593 | — | Legacy adapter; duplicated mini core removed |
| `src/visualization/__init__.py` | 0 | — | Retained optional workflow / CLI |
| `src/visualization/cluster_center_patches.py` | 342 | src.visualization.cluster_patches | Retained optional workflow / CLI |
| `src/visualization/cluster_patches.py` | 568 | — | Retained optional workflow / CLI |
| `src/visualization/embedding_2d.py` | 225 | — | Retained optional workflow / CLI |
| `src/visualization/image_cls_attention.py` | 194 | src.features.dinov2, src.features.patch_store | Retained optional workflow / CLI |
| `src/visualization/image_local_centers_attention.py` | 179 | src.features.patch_store, src.visualization.cluster_patches | Retained optional workflow / CLI |
