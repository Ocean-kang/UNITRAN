# UNITRAN

UNITRAN is a lightweight research codebase for studying unpaired translation and distribution alignment between DINOv2 vision features and text embeddings.

The current repository keeps the workflow explicit and simple: `tools/` contains command-line entrypoints, `unitran/` contains reusable implementations, `scripts/` contains runnable shell examples, and `third_party/` stores vendored external code.

## What this repository does

The main experimental flow is:

```text
COCO images
  -> extract DINOv2 patch tokens
  -> cluster patch tokens globally or per image
  -> inspect clusters with query and visualization tools
  -> build / cluster text embeddings or text atoms
  -> compare text and vision distributions
  -> train a simple UNITRAN mapping between embedding spaces
```

Supported tasks:

1. Extract DINOv2 patch tokens from MS COCO images.
2. Run streaming KMeans over all patch tokens.
3. Build per-image local patch centers and optionally cluster them globally.
4. Query COCO captions or patch-token cluster assignments for one image.
5. Visualize cluster top-k patches, cluster extremes, CLS attention, local-center attention, and 2D embedding projections.
6. Cluster caption/text embeddings.
7. Evaluate text/vision distribution density with original captions and LLM-expanded semantic atoms.
8. Train a simple unpaired UNITRAN-style orthogonal mapping.

## Repository structure

```text
UNITRAN/
├── README.md
├── conda.yaml
├── scripts/
│   ├── extract/
│   ├── cluster/
│   ├── query/
│   ├── visualize/
│   └── evaluate/
├── tools/
│   ├── extract.py
│   ├── cluster.py
│   ├── query.py
│   ├── visualize.py
│   ├── evaluate.py
│   ├── generate_text_atoms.py
│   └── train_unitran.py
├── unitran/
│   ├── datasets/
│   ├── features/
│   ├── clustering/
│   ├── visualization/
│   ├── evaluation/
│   ├── embeddings/
│   ├── training/
│   └── utils/
└── third_party/
    └── llama/
```

The preferred entrypoints are grouped by task:

```bash
python tools/extract.py --help
python tools/cluster.py <patch-global|patch-local|caption|match-centroids> --help
python tools/query.py <coco-captions|image-clusters> --help
python tools/visualize.py <cluster-topk|cluster-extremes|image-cls-attn|image-local-centers|embeddings2d> --help
python tools/evaluate.py <distribution|text-density> --help
python tools/generate_text_atoms.py --help
python tools/train_unitran.py --help
```

## Environment

Create the environment from the repository root:

```bash
conda env create -f conda.yaml
conda activate unitran
```

The main dependencies are Python 3.10, PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, tqdm, and Pillow. DINOv2 is loaded through `torch.hub`, so the model must be downloadable or already cached on the server.

## Data layout

For the COCO patch-token workflow, the default layout is:

```text
coco/
├── train2014/
│   └── COCO_train2014_*.jpg
└── annotations/
    └── captions_train2014.json
```

The default feature directory is:

```text
feature/coco2014_dinov2_vitb14_448/
```

After extraction, it contains:

```text
feature/coco2014_dinov2_vitb14_448/
├── images.jsonl
├── meta.json
├── patch_tokens_shape.json
└── patch_tokens/
    ├── shard_000_fp16.mmap
    └── shard_000.meta.json
```

## 1. Extract DINOv2 patch tokens

Run the multi-GPU launcher:

```bash
bash scripts/extract/run_extract_coco_dinov2_patch_4gpu.sh
```

Single-process example:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/extract.py \
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

Use `--save_image_mean_pt` if you also want image-level mean patch features for embedding experiments.

## 2. Cluster vision patch tokens

### Global patch-token KMeans

```bash
bash scripts/cluster/run_kmeans_coco_dinov2_patch.sh
```

Equivalent command:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/cluster.py patch-global \
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

Increase `--chunk_images` for better throughput if GPU memory allows. Use smaller `--k`, fewer `--num_iters`, and `--max_images` for debugging.

### Per-image local centers

```bash
bash scripts/cluster/run_local_patch_center_kmeans_coco_dinov2.sh
```

This compresses each image's raw patch tokens into `LOCAL_K` local centers, then optionally runs global KMeans on the local-center feature directory.

## 3. Query images and clusters

Query COCO captions:

```bash
python tools/query.py coco-captions \
  --coco_root ./coco \
  --split train2014 \
  --image_name COCO_train2014_000000000009.jpg
```

Query the cluster-id grid for one image:

```bash
python tools/query.py image-clusters \
  --feature_dir feature/coco2014_dinov2_vitb14_448 \
  --image_name COCO_train2014_000000000009.jpg \
  --ks all
```

The image-cluster query writes outputs under:

```text
feature/coco2014_dinov2_vitb14_448/query_image_patch_clusters/<image_stem>/
```

## 4. Visualize vision clusters

Visualize top-k patches for selected global clusters:

```bash
python tools/visualize.py cluster-topk \
  --feature_dir feature/coco2014_dinov2_vitb14_448 \
  --k 1024 \
  --topk 50 \
  --clusters 0 \
  --metric cosine \
  --chunk_images 1024 \
  --tile_size 224 \
  --collage_cols 5
```

Visualize the most similar and least similar assigned patches within a cluster:

```bash
python tools/visualize.py cluster-extremes \
  --feature_dir feature/coco2014_dinov2_vitb14_448 \
  --k 1024 \
  --topk 50 \
  --clusters 0 \
  --metric cosine
```

Visualize DINOv2 CLS attention for one image:

```bash
python tools/visualize.py image-cls-attn \
  --feature_dir feature/coco2014_dinov2_vitb14_448 \
  --image_name COCO_train2014_000000000009.jpg \
  --device cuda
```

Visualize per-image local-center attention maps:

```bash
python tools/visualize.py image-local-centers \
  --raw_feature_dir feature/coco2014_dinov2_vitb14_448 \
  --local_feature_dir feature/coco2014_dinov2_vitb14_448_local_k20 \
  --image_name COCO_train2014_000000000009.jpg \
  --local_centers all \
  --metric cosine
```

## 5. Cluster and visualize text embeddings

Cluster existing text embeddings:

```bash
python tools/cluster.py caption \
  --input_pt feature/text_embeddings.pt \
  --text_key text_feats \
  --k 512 \
  --preprocess l2 \
  --out_dir feature/caption_kmeans \
  --overwrite
```

Visualize vision and text centroids in one 2D space:

```bash
python tools/visualize.py embeddings2d \
  --vision_pt feature/coco2014_dinov2_vitb14_448/kmeans/centroids_k512_fp32.npy \
  --text_pt feature/caption_kmeans/centroids_k512_fp32.npy \
  --method pca \
  --preprocess l2 \
  --out_dir outputs/vis \
  --prefix vision_caption_centroids_k512
```

If dimensions differ, the tool preserves the existing zero-padding behavior before projection.

## 6. Text-density evaluation with semantic atoms

The text-density pipeline tests whether expanded text atoms better match the DINOv2 patch-token distribution than original COCO captions.

Run the launcher:

```bash
bash scripts/evaluate/run_text_density_eval_1000.sh
```

Or run the three stages manually:

```bash
python tools/evaluate.py text-density sample \
  --coco_root ./coco \
  --split train2014 \
  --feature_dir feature/coco2014_dinov2_vitb14_448 \
  --num_images 1000 \
  --out_jsonl outputs/text_density_eval/sample_1000.jsonl
```

```bash
python tools/generate_text_atoms.py \
  --sample_jsonl outputs/text_density_eval/sample_1000.jsonl \
  --out_jsonl outputs/text_density_eval/recaption_atoms_1000.jsonl \
  --llama_root ./third_party \
  --ckpt_dir /path/to/Meta-Llama-3-8B-Instruct \
  --tokenizer_path /path/to/Meta-Llama-3-8B-Instruct/tokenizer.model \
  --max_batch_size 4 \
  --resume
```

```bash
python tools/evaluate.py text-density features \
  --sample_jsonl outputs/text_density_eval/sample_1000.jsonl \
  --recaption_jsonl outputs/text_density_eval/recaption_atoms_1000.jsonl \
  --feature_dir feature/coco2014_dinov2_vitb14_448 \
  --text_model sentence-transformers/all-mpnet-base-v2 \
  --patches_per_image 64 \
  --device cuda \
  --out_dir outputs/text_density_eval

python tools/evaluate.py text-density report \
  --out_dir outputs/text_density_eval \
  --max_points 20000
```

Main outputs:

```text
outputs/text_density_eval/
├── sample_1000.jsonl
├── recaption_atoms_1000.jsonl
├── density_eval_original.pt
├── density_eval_expanded.pt
└── density_report.json
```

## 7. Train UNITRAN mapping

`tools/train_unitran.py` expects the embedding directory to contain:

```text
talk2dino_avg_self_attn_out_80000.pt
val_paired_8192.pt
```

Each file should contain `vision_feats` and `text_feats`.

Run:

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

## 8. General distribution metrics

For one file containing both `vision_feats` and `text_feats`:

```bash
python tools/evaluate.py distribution \
  --input_pt /path/to/features.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --preprocess unitran \
  --max_points 2000 \
  --out_json outputs/distribution_metric.json
```

For two separate files:

```bash
python tools/evaluate.py distribution \
  --vision_pt /path/to/vision.pt \
  --text_pt /path/to/text.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --preprocess unitran \
  --out_json outputs/distribution_metric.json
```

## Smoke checks

Run these checks after installing dependencies:

```bash
python -m compileall -q tools unitran third_party

python tools/extract.py --help
python tools/cluster.py patch-global --help
python tools/cluster.py patch-local --help
python tools/cluster.py caption --help
python tools/cluster.py match-centroids --help
python tools/query.py coco-captions --help
python tools/query.py image-clusters --help
python tools/visualize.py cluster-topk --help
python tools/visualize.py cluster-extremes --help
python tools/visualize.py image-cls-attn --help
python tools/visualize.py image-local-centers --help
python tools/visualize.py embeddings2d --help
python tools/evaluate.py distribution --help
python tools/evaluate.py text-density sample --help
python tools/evaluate.py text-density features --help
python tools/evaluate.py text-density report --help
python tools/generate_text_atoms.py --help
python tools/train_unitran.py --help
```

Full extraction, clustering, and visualization checks require COCO files, DINOv2 weights, feature files, and a CUDA environment.

## Common issues

### Script cannot find `tools/...`

Run shell launchers from the repository root, or use the launchers under `scripts/<workflow>/`, which change to the repository root automatically.

### Visualization cannot find centroid or assignment files

Make sure `--k` matches the KMeans run. For example, `--k 512` produces `centroids_k512_fp32.npy`, not `centroids_k1024_fp32.npy`.

### KMeans is slow

Streaming KMeans scans all patch tokens every iteration. Increase `--chunk_images` if memory allows, reduce `--max_images` for debugging, and test with smaller `--k` before full runs.

### DINOv2 model loading fails

Make sure the server can access `torch.hub` or already has the DINOv2 model cached.

## License

See `LICENSE`.
