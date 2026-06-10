# UNITRAN

**UNITRAN** is a compact research codebase for building DINOv2 vision features, LLM-generated text atoms, density evaluation banks, and unpaired vision-text mapping models.

The maintained workflow is config-driven: use `experiments/*.py` with `configs/*.yaml`. Core implementations live in `src/`; generated features and experiment outputs are kept outside version control.

## Method Overview

UNITRAN follows a lightweight vision-text distribution alignment pipeline:

1. Extract DINOv2 patch tokens from COCO images.
2. Build global patch clusters or per-image local patch centers.
3. Generate text atoms from COCO captions with an LLM backend.
4. Encode original captions and expanded text atoms into text features.
5. Compare vision/text feature distributions with density metrics.
6. Train a mini-vec2vec / vec2vec-style unpaired mapping between feature spaces.

The repository does not modify COCO annotations or pretrained model weights.

## Repository Structure

```text
configs/        Maintained YAML configs for the supported workflows
experiments/    Recommended runnable Python entry points
src/            Implementation modules for datasets, features, clustering, evaluation, training, and visualization
scripts/        Thin shell wrappers around the recommended experiment commands
external/       Optional Llama-compatible code used by the text-atom backend
docs/           Additional workflow notes and output-format descriptions
```

Runtime directories are ignored by git:

```text
data/           User-prepared datasets, e.g. MSCOCO2014 / MSCOCO2017
feature/        Reusable intermediate features: patch tokens, local centers, text embeddings, prototypes
outputs/        Experiment reports, visualization images, mapping checkpoints, and logs
```

## Installation

```bash
conda env create -f conda.yaml
conda activate unitran
```

Install any extra packages required by your selected vision or text backend, such as `torch`, `torchvision`, `sentence-transformers`, `transformers`, and DINOv2 dependencies.

## Dataset Preparation

Prepare COCO data under `data/`:

```text
data/mscoco14/
├── annotations/
├── train2014/
└── val2014/

data/mscoco17/
├── annotations/
├── train2017/
├── val2017/
├── panoptic_train2017/
└── panoptic_val2017/
```

Most configs default to `mscoco14`. Switch to COCO2017 with:

```bash
--set active_dataset=mscoco17
```

## Feature Extraction

Extract DINOv2 patch tokens:

```bash
python experiments/extract_dinov2_features.py --config configs/extract_dinov2.yaml
```

Default output:

```text
feature/vision/coco2014_dinov2_vitb14_reg_448/
```

For COCO2017:

```bash
python experiments/extract_dinov2_features.py \
  --config configs/extract_dinov2.yaml \
  --set active_dataset=mscoco17
```

## Clustering

Run global patch-token clustering:

```bash
python experiments/run_global_patch_cluster.py --config configs/cluster_global.yaml
```

Run per-image local patch clustering:

```bash
python experiments/run_local_patch_cluster.py --config configs/cluster_local.yaml
```

Default local-center output:

```text
feature/vision/coco2014_dinov2_vitb14_reg_448_local_k20/
```

## Text Atom Generation

Generate structured text atoms from sampled COCO captions:

```bash
python experiments/run_text_atoms.py --config configs/text_atoms.yaml
```

Default output:

```text
feature/text/coco2014_llama_atoms_k20/text_atoms.jsonl
```

The default config assumes a local HuggingFace-style Llama directory:

```text
Meta-Llama-3-8B-Instruct-HF/
```

Edit `configs/text_atoms.yaml` when your model path or backend differs.

## Density Evaluation

Build local-center/text feature banks and write density reports:

```bash
python experiments/run_density_eval.py --config configs/density_local.yaml --stage all
```

Stages can also be run separately:

```bash
python experiments/run_density_eval.py --config configs/density_local.yaml --stage sample
python experiments/run_density_eval.py --config configs/density_local.yaml --stage features
python experiments/run_density_eval.py --config configs/density_local.yaml --stage report
```

Reusable feature banks are written to:

```text
feature/density/coco2014_n1000_k20/
```

Evaluation reports are written to:

```text
outputs/density/coco2014_n1000_k20/
```

For a different sample size, override both the sample count and output directories:

```bash
python experiments/run_density_eval.py \
  --config configs/density_local.yaml \
  --stage all \
  --set data.num_images=5000 \
  --set feature.out_dir=feature/density/coco2014_n5000_k20 \
  --set experiment.output_dir=outputs/density/coco2014_n5000_k20
```

## Training

Train the mapping model:

```bash
python experiments/train_mapping.py --config configs/train_mapping.yaml
```

The mapping method is selected in the config:

```yaml
mapping:
  method: mini_vec2vec   # or vec2vec
```

Default outputs:

```text
outputs/mapping/coco2014_n1000_k20/
```

## Evaluation

Extract class-level evaluation prototypes from COCO masks:

```bash
python experiments/extract_eval_prototypes.py --config configs/eval_prototypes.yaml
```

Default output for COCO2017:

```text
feature/eval/mscoco2017/mscoco2017_eval_prototypes.pt
```

Train mapping with an eval prototype file:

```bash
python experiments/train_mapping.py --config configs/train_mapping_eval.yaml
```

## Visualization

Run the default cluster visualization:

```bash
python experiments/run_visualize_cluster.py --config configs/visualize_cluster.yaml
```

Supported visualization modes are configured through `visualization.mode`:

```text
cluster_topk
cluster_extremes
image_cls_attention
image_local_centers
embedding_2d
```

Default visualization outputs:

```text
outputs/visualize/
```

## Output Directory Policy

UNITRAN separates reusable intermediate features from experiment results:

```text
feature/vision/     DINOv2 patch tokens, global clusters, local centers
feature/text/       LLM-generated text atoms and reusable text-side intermediates
feature/density/    Density-eval feature banks and sampled image lists
feature/eval/       Class-level eval prototypes
outputs/density/    Density reports and resolved configs
outputs/visualize/  Visualization images and metadata
outputs/mapping/    Mapping checkpoints and metrics
```

## Scripts

Shell wrappers are provided for the maintained flow:

```bash
bash scripts/run_1_extract.sh
bash scripts/run_2_global_cluster.sh
bash scripts/run_2_local_cluster.sh
bash scripts/run_3_text_atoms.sh
bash scripts/run_4_density_eval.sh
bash scripts/run_5_visualize.sh
```

Run a lightweight sanity check:

```bash
bash scripts/smoke_check.sh
```

## Common Issues

**Where should new results be saved?**  
Save reusable feature banks under `feature/` and experiment reports, visualizations, logs, or checkpoints under `outputs/`.

**How do I switch from COCO2014 to COCO2017?**  
Use the same maintained config and override `active_dataset`, for example `--set active_dataset=mscoco17`.

**Do I need to regenerate text atoms during every density evaluation?**  
No. Generate `text_atoms.jsonl` once, then let density evaluation read it from `feature/text/...`.

**Are distribution metrics enough to prove semantic alignment?**  
No. MMD, SWD, and coverage-style metrics describe geometric distribution differences. Use them together with retrieval/eval prototypes and visualization.

## Citation

If this repository is used in a paper, add the project citation here.

```bibtex
@misc{unitran,
  title  = {UNITRAN: Unpaired Vision-Text Distribution Alignment},
  author = {Anonymous},
  year   = {2026}
}
```

## Acknowledgement

This repository builds on ideas and tooling from DINOv2, MSCOCO, Llama / HuggingFace text generation, sentence-transformers, and vec2vec-style unpaired mapping methods.
