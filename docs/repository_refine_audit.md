# UNITRAN repository refine audit

This document records the repository audit and the minimal structure refine performed in this pass.

## 1. Original repository structure

Original files at the repository root:

```text
UNITRAN-main/
├── README.md
├── LICENSE
├── conda.yaml
├── .gitignore
├── main.py
├── distribution_metric.py
├── visualize_embeddings_2d.py
├── extract_coco_dinov2_patch.py
├── cluster_coco_dinov2_streaming.py
├── visualize_cluster_topk_patches.py
├── kmeans.py
├── run_extract_coco_dinov2_patch_4gpu.sh
├── run_kmeans_coco_dinov2_patch.sh
└── run_visualize_cluster_topk_patches.sh
```

### File responsibilities

- `main.py`: trains an unsupervised orthogonal mapping from vision embeddings to text embeddings using clustering, QAP-based centroid alignment, pseudo matching, Orthogonal Procrustes, nearest-neighbor refinement, and final cluster-based correction.
- `distribution_metric.py`: computes distribution-level metrics between vision and text embeddings, including cosine statistics, MMD, KMeans-center Wasserstein distance, coverage precision/recall, Spearman structure consistency, and entropic GW distance.
- `visualize_embeddings_2d.py`: visualizes text/vision embeddings with PCA or t-SNE, optionally before and after applying a learned `W`.
- `extract_coco_dinov2_patch.py`: extracts COCO DINOv2 patch tokens into memory-mapped shard files and optional image-level mean features.
- `cluster_coco_dinov2_streaming.py`: performs streaming full-batch KMeans over extracted patch-token shards and optionally saves per-patch cluster assignments/distances.
- `visualize_cluster_topk_patches.py`: finds top-k patch tokens per cluster, maps them back to original images, draws selected patches, generates centroid attention overlays, and saves collages.
- `kmeans.py`: standalone FAISS KMeans helper. It is not referenced by the default scripts.
- `run_*.sh`: convenience scripts for feature extraction, KMeans clustering, and cluster top-k visualization.

## 2. Existing functionality

The current repository genuinely supports these workflows:

1. UNITRAN mapping training from precomputed paired/unpaired embedding tensors.
2. Distribution metric evaluation between vision/text embedding sets.
3. 2D embedding visualization with PCA or t-SNE.
4. COCO 2014 DINOv2 patch-token extraction.
5. Streaming KMeans clustering over extracted patch tokens.
6. Cluster top-k patch visualization and centroid attention overlays.
7. A retained FAISS KMeans helper module for manual reuse.

No new model, dataset, metric, visualization type, training framework, or experiment task was introduced in this refine.

## 3. Main issues found

### Directory structure

All executable Python files, shell scripts, and helper modules were placed at the repository root. This made the root directory harder to scan and blurred the difference between CLI tools, reusable code, and documentation.

### File naming and entry points

`main.py` was not descriptive. For a research code release, `train_unitran.py` is clearer because it states the actual workflow. Shell scripts were correctly named but were mixed with source files at the root.

### Parameter and path management

Most scripts already expose important paths through `argparse`, which is good. The main remaining issue is documentation-level clarity rather than a need for a complex configuration framework. The default feature/output paths are preserved.

### Output organization

Generated data are already ignored through `.gitignore` (`/coco`, `/feature`, `/outputs`). The feature extraction and clustering scripts use a coherent output layout under `feature/coco2014_dinov2_vitb14_448/`.

### Import organization

The default CLI scripts do not rely on fragile intra-repository imports. This makes file movement low risk. The FAISS helper was standalone and therefore better placed in a lightweight package namespace.

### README and reproducibility

The original README documented the early UNITRAN mapping/metric/embedding visualization scripts, but did not fully cover the later COCO patch-token extraction, streaming KMeans, or cluster visualization scripts. It also still referenced root-level script paths.

## 4. Minimal refine plan applied

The applied structure keeps the project lightweight:

- Keep project metadata at root: `README.md`, `LICENSE`, `.gitignore`, `conda.yaml`.
- Move runnable Python entry points to `tools/`.
- Move runnable shell scripts to `scripts/`.
- Move the reusable FAISS helper to `unitran/clustering/faiss_kmeans.py`.
- Add only minimal `__init__.py` files for the package namespace.
- Add this audit document under `docs/`.
- Rewrite README to describe only the workflows currently supported by the repository.

## 5. Files moved or renamed

```text
main.py                                  -> tools/train_unitran.py
distribution_metric.py                   -> tools/evaluate_distribution.py
visualize_embeddings_2d.py               -> tools/visualize_embeddings_2d.py
extract_coco_dinov2_patch.py             -> tools/extract_coco_dinov2_patch.py
cluster_coco_dinov2_streaming.py         -> tools/cluster_coco_dinov2_streaming.py
visualize_cluster_topk_patches.py        -> tools/visualize_cluster_topk_patches.py
kmeans.py                                -> unitran/clustering/faiss_kmeans.py
run_extract_coco_dinov2_patch_4gpu.sh    -> scripts/run_extract_coco_dinov2_patch_4gpu.sh
run_kmeans_coco_dinov2_patch.sh          -> scripts/run_kmeans_coco_dinov2_patch.sh
run_visualize_cluster_topk_patches.sh    -> scripts/run_visualize_cluster_topk_patches.sh
```

## 6. Risk analysis

### Script path changes

Risk: old commands like `python main.py` no longer point to the training entry point.

Mitigation: README and shell scripts were updated to use `python tools/train_unitran.py` and other new `tools/` paths.

### Shell script path changes

Risk: shell scripts that call root-level Python files would fail after the move.

Mitigation: all shell scripts in `scripts/` were updated to call `python tools/...` from the repository root.

### Documentation drift

Risk: README may describe features not actually implemented.

Mitigation: README was rewritten from the inspected files and only includes the current scripts and their existing command-line options/workflows.

### Imports

Risk: moving files can break local imports.

Mitigation: the default CLI scripts do not import each other. The moved FAISS helper is not imported by the default tools, so moving it does not affect the main workflows.

### Data/output paths

Risk: changing default output paths would break existing workflows.

Mitigation: default paths such as `./coco`, `feature/coco2014_dinov2_vitb14_448`, and `outputs/` were preserved.

## 7. Validation performed

A syntax compilation check was run over the refined repository:

```bash
python -m py_compile \
  tools/train_unitran.py \
  tools/evaluate_distribution.py \
  tools/visualize_embeddings_2d.py \
  tools/extract_coco_dinov2_patch.py \
  tools/cluster_coco_dinov2_streaming.py \
  tools/visualize_cluster_topk_patches.py \
  unitran/clustering/faiss_kmeans.py
```

This verifies syntax and path-level refactor consistency. Full data-dependent runs require the actual COCO data, DINOv2 checkpoint/cache, precomputed embeddings, and GPU environment.

## 8. Deliberately not changed

- No new model was added.
- No new dataset support was added.
- No algorithm logic was rewritten.
- No new metric or visualization type was added.
- No Hydra/Lightning/WandB/Docker/CI/pre-commit framework was introduced.
- No large unit-test system was added.
- Existing default output paths were preserved.
