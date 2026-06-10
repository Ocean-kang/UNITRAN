# UNITRAN

**UNITRAN** is a research codebase for building DINOv2 vision features, LLM-generated text atoms, density evaluation features, and unpaired vision-text mapping models.

The repository is organized around config-driven experiment entry points. Core model and experiment logic is kept in `src/`, while runnable workflows are exposed through `experiments/` and `configs/`.

## Method Overview

UNITRAN follows a lightweight vision-text distribution alignment workflow:

1. Extract DINOv2 patch tokens from COCO images.
2. Build global patch clusters or per-image local patch centers.
3. Generate text atoms from COCO captions with an LLM backend.
4. Encode original captions and expanded text atoms into text features.
5. Compare vision/text feature distributions with density metrics.
6. Train a UNITRAN / vec2vec-style unpaired mapping between feature spaces.

The code does not change COCO annotations or pretrained model weights. Generated features and experiment outputs are written outside version control.

## Repository Structure

```text
configs/        YAML configs for feature extraction, clustering, text atoms, evaluation, visualization, and mapping
experiments/    Recommended runnable Python entry points
src/            Implementation modules for datasets, features, clustering, evaluation, training, and visualization
scripts/        Thin shell wrappers for the recommended experiment commands
external/       Optional external Llama-compatible code
/docs           Additional notes on workflow, result formats, and known issues
```

Runtime directories are intentionally ignored by git:

```text
data/           User-prepared datasets, e.g. MSCOCO2014 / MSCOCO2017
feature/        Reusable intermediate features: patch tokens, local centers, text embeddings, prototypes
outputs/        Experiment reports, visualizations, mapping checkpoints, and logs
```

## Installation

Create the conda environment:

```bash
conda env create -f conda.yaml
conda activate unitran
```

Install any extra packages required by your selected backbone or text model, such as `torch`, `torchvision`, `sentence-transformers`, `transformers`, and DINOv2 dependencies, if they are not already available in your environment.

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

The default configs use `mscoco14`. To switch a supported workflow to COCO2017, use:

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

Edit `configs/text_atoms.yaml` if your model path or backend differs.

## Density Evaluation

Build local-center/text feature banks and write density reports:

```bash
python experiments/run_density_eval.py --config configs/density_local.yaml --stage all
```

Stages can be run separately:

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

Shell wrappers are provided for the recommended flow:

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

**Dataset path not found.** Check that COCO files are placed under `data/mscoco14` or `data/mscoco17`, or override paths with `--set`.

**Llama model path not found.** Update `configs/text_atoms.yaml` to match your local model path.

**CUDA out of memory during clustering.** Reduce `cluster.chunk_images` or `runtime.chunk_images` in the relevant config.

**Feature and report paths look different.** This is expected: reusable tensors go to `feature/`, while reports, figures, and checkpoints go to `outputs/`.

## Citation

If you use this repository, please cite the corresponding paper or project once available.

```bibtex
@misc{unitran2026,
  title  = {UNITRAN: Unpaired Vision-Text Feature Mapping with Dense Vision and Text Features},
  author = {Anonymous},
  year   = {2026}
}
```

## Acknowledgement

This project builds on pretrained vision-language and self-supervised vision models, including DINOv2, COCO, HuggingFace Transformers / SentenceTransformers, and Llama-compatible generation code.
