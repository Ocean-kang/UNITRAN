# UNITRAN

UNITRAN is a lightweight research codebase for unpaired translation/alignment between vision and text embedding spaces. The current repository also contains the COCO DINOv2 patch-token extraction, streaming patch-token clustering, and cluster top-k visualization utilities used in the current experiments.

This repository is intentionally kept small: it uses plain Python entry scripts, shell launchers, and explicit command-line arguments instead of a heavy experiment framework.

## Current supported functionality

The repository currently supports the following workflows:

1. Train a UNITRAN-style orthogonal mapping `W` from vision embeddings to text embeddings using unpaired training splits and paired validation.
2. Compute distribution-level metrics between vision and text embeddings.
3. Visualize text/vision embeddings in 2D with PCA or t-SNE, optionally before and after applying a learned `W`.
4. Extract MS COCO 2014 DINOv2 patch tokens into sharded memory-mapped files.
5. Run streaming full-batch KMeans over extracted patch tokens.
6. Visualize top-k patch tokens for selected KMeans clusters and save patch/attention collages.

No additional model architecture, dataset, training framework, or evaluation metric is introduced by the current codebase.

## Repository structure

```text
UNITRAN/
├── README.md
├── LICENSE
├── conda.yaml
├── .gitignore
├── docs/
│   └── repository_refine_audit.md
├── scripts/
│   ├── run_extract_coco_dinov2_patch_4gpu.sh
│   ├── run_kmeans_coco_dinov2_patch.sh
│   └── run_visualize_cluster_topk_patches.sh
├── tools/
│   ├── train_unitran.py
│   ├── evaluate_distribution.py
│   ├── visualize_embeddings_2d.py
│   ├── extract_coco_dinov2_patch.py
│   ├── cluster_coco_dinov2_streaming.py
│   └── visualize_cluster_topk_patches.py
└── unitran/
    ├── __init__.py
    └── clustering/
        ├── __init__.py
        └── faiss_kmeans.py
```

### Main files

- `tools/train_unitran.py`: trains the vision-to-text orthogonal mapping `W`.
- `tools/evaluate_distribution.py`: computes distribution metrics between two embedding sets.
- `tools/visualize_embeddings_2d.py`: visualizes embedding spaces with PCA or t-SNE.
- `tools/extract_coco_dinov2_patch.py`: extracts DINOv2 patch tokens from COCO images.
- `tools/cluster_coco_dinov2_streaming.py`: runs streaming KMeans over patch-token shards.
- `tools/visualize_cluster_topk_patches.py`: visualizes top-k patches for selected clusters.
- `unitran/clustering/faiss_kmeans.py`: retained FAISS KMeans helper; not used by the default scripts.
- `scripts/*.sh`: reproducible shell launchers for the current COCO patch-token workflow.
- `docs/repository_refine_audit.md`: audit and structure-refine notes.

## Environment

Recommended environment:

- Python 3.10
- PyTorch with CUDA support
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- tqdm
- Pillow

Create the conda environment from the repository root:

```bash
conda env create -f conda.yaml
conda activate unitran
```

Check the environment:

```bash
python - <<'PY'
import torch
import numpy as np
import scipy
import sklearn
import matplotlib
import tqdm
from PIL import Image

print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda runtime:', torch.version.cuda)
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

## Data preparation

### COCO patch-token workflow

The extraction script expects the COCO 2014 directory by default under `./coco`:

```text
coco/
├── train2014/
│   ├── COCO_train2014_000000000009.jpg
│   └── ...
└── annotations/
    └── captions_train2014.json   # optional but recommended
```

The generated feature directory is ignored by git and defaults to:

```text
feature/coco2014_dinov2_vitb14_448/
```

### UNITRAN embedding workflow

`tools/train_unitran.py` currently expects `--embedding_dir` to contain these two files:

```text
talk2dino_avg_self_attn_out_80000.pt
val_paired_8192.pt
```

Each file should contain:

```text
vision_feats
text_feats
```

The file names are still hard-coded in the current training script. This is preserved to avoid changing behavior.

## Workflow 1: extract COCO DINOv2 patch tokens

Run the provided 4-GPU launcher from the repository root:

```bash
bash scripts/run_extract_coco_dinov2_patch_4gpu.sh
```

Equivalent single-command form:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/extract_coco_dinov2_patch.py \
  --coco_root ./coco \
  --split train2014 \
  --out_dir feature/coco2014_dinov2_vitb14_448 \
  --model dinov2_vitb14 \
  --image_size 448 \
  --batch_size 32 \
  --num_workers 8 \
  --save_dtype float16 \
  --amp_dtype float16 \
  --write_global_meta
```

Important outputs:

```text
feature/coco2014_dinov2_vitb14_448/
├── images.jsonl
├── meta.json
├── patch_tokens_shape.json
└── patch_tokens/
    ├── shard_000_fp16.mmap
    └── shard_000.meta.json
```

If `--save_image_mean_pt` is set, the script can also save image-level mean patch features for UNITRAN-style embedding experiments.

## Workflow 2: stream KMeans over patch tokens

Run the launcher:

```bash
bash scripts/run_kmeans_coco_dinov2_patch.sh
```

Equivalent command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/cluster_coco_dinov2_streaming.py \
  --feature_dir feature/coco2014_dinov2_vitb14_448 \
  --k 512 \
  --num_iters 20 \
  --metric cosine \
  --chunk_images 128 \
  --compute_dtype float16 \
  --device cuda:0 \
  --assign \
  --save_dist \
  --overwrite
```

Important outputs:

```text
feature/coco2014_dinov2_vitb14_448/
├── kmeans/
│   ├── centroids_k512_fp32.npy
│   ├── centroids_k512_fp16.npy
│   ├── cluster_counts_k512.npy
│   ├── config_k512.json
│   └── train_log_k512.jsonl
└── assignment/
    ├── patch_cluster_ids_k512_uint16.npy
    ├── patch_cluster_ids_k512_shape.json
    └── patch_cluster_dist_k512_fp16.npy
```

Use `--chunk_images` to control the number of images streamed per chunk. Larger values usually improve throughput but increase GPU memory use.

## Workflow 3: visualize cluster top-k patches

Run the launcher:

```bash
bash scripts/run_visualize_cluster_topk_patches.sh
```

Equivalent command for one cluster:

```bash
python tools/visualize_cluster_topk_patches.py \
  --feature_dir feature/coco2014_dinov2_vitb14_448 \
  --k 1024 \
  --topk 50 \
  --clusters 0 \
  --metric cosine \
  --chunk_images 1024 \
  --tile_size 224 \
  --collage_cols 5
```

The script expects the corresponding centroid and assignment files under:

```text
feature/coco2014_dinov2_vitb14_448/kmeans/
feature/coco2014_dinov2_vitb14_448/assignment/
```

For example, if clustering was run with `--k 512`, visualization should also use `--k 512`.

## Workflow 4: train UNITRAN mapping

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_unitran.py \
  --embedding_dir /path/to/embedding_dir \
  --source dinov2 \
  --target text \
  --out_dir outputs/unitran \
  --seed 0
```

Main outputs:

```text
outputs/unitran/
├── W_dinov2_to_text_seed0.pt
└── result_dinov2_to_text_seed0.json
```

## Workflow 5: compute distribution metrics

If one `.pt` file contains both `vision_feats` and `text_feats`:

```bash
python tools/evaluate_distribution.py \
  --input_pt /path/to/features.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --preprocess unitran \
  --max_points 2000 \
  --out_json outputs/distribution_metric.json
```

If vision/text features are stored in two files:

```bash
python tools/evaluate_distribution.py \
  --vision_pt /path/to/vision.pt \
  --text_pt /path/to/text.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --preprocess unitran \
  --out_json outputs/distribution_metric.json
```

## Workflow 6: visualize embedding spaces

Before applying `W`:

```bash
python tools/visualize_embeddings_2d.py \
  --input_pt /path/to/features.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --method pca \
  --preprocess unitran \
  --out_dir outputs/vis
```

Before and after applying a learned `W`:

```bash
python tools/visualize_embeddings_2d.py \
  --input_pt /path/to/features.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --W outputs/unitran/W_dinov2_to_text_seed0.pt \
  --method pca \
  --preprocess unitran \
  --out_dir outputs/vis
```

## Smoke checks

Run these commands from the repository root after installing dependencies:

```bash
python -m py_compile \
  tools/train_unitran.py \
  tools/evaluate_distribution.py \
  tools/visualize_embeddings_2d.py \
  tools/extract_coco_dinov2_patch.py \
  tools/cluster_coco_dinov2_streaming.py \
  tools/visualize_cluster_topk_patches.py \
  unitran/clustering/faiss_kmeans.py

python tools/train_unitran.py --help
python tools/evaluate_distribution.py --help
python tools/visualize_embeddings_2d.py --help
python tools/extract_coco_dinov2_patch.py --help
python tools/cluster_coco_dinov2_streaming.py --help
python tools/visualize_cluster_topk_patches.py --help
```

Full workflow checks require the relevant data files and GPU environment.

## Common issues

### `tools/train_unitran.py` cannot find embedding files

The current training script expects fixed file names inside `--embedding_dir`:

```text
talk2dino_avg_self_attn_out_80000.pt
val_paired_8192.pt
```

Rename or symlink your feature files accordingly, or modify the script explicitly if you want configurable train/validation file names.

### Visualization cannot find `centroids_k*.npy` or assignment files

Make sure `--k` in `tools/visualize_cluster_topk_patches.py` matches the `--k` used during KMeans clustering. For example, KMeans with `--k 512` produces `centroids_k512_fp32.npy`, not `centroids_k1024_fp32.npy`.

### KMeans is slow

The streaming KMeans script processes every patch token each iteration. Increase `--chunk_images` if GPU memory allows, reduce `--max_images` for debugging, and run quick sanity checks with smaller `--k` and fewer `--num_iters` before full runs.

### DINOv2 model loading fails

`tools/extract_coco_dinov2_patch.py` uses `torch.hub` to load DINOv2. Make sure the server can access the model source or has the required model cached.

## Citation

TODO: add paper citation when available.

## Acknowledgement

TODO: add acknowledgements for upstream codebases, pretrained models, or datasets as appropriate.

## License

See `LICENSE`.
