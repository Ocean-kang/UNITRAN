# Result Format

This document records the main files produced by the recommended config-driven pipeline.

## DINOv2 Feature Output

```text
feature/vision/coco2014_dinov2_vitb14_reg_448/
├── meta.json
├── images.jsonl
├── patch_tokens_shape.json
├── resolved_config.yaml
└── patch_tokens/
    ├── shard_000_fp16.mmap
    └── shard_000.meta.json
```

`patch_tokens/` stores reusable DINOv2 patch-token features. The exact shard names depend on the shard configuration.

## Global Clustering Output

```text
feature/vision/coco2014_dinov2_vitb14_reg_448/
├── kmeans/
│   ├── centroids_k1024_fp32.npy
│   ├── centroids_k1024_fp16.npy
│   └── train_log_k1024.jsonl
└── assignment/
    ├── patch_cluster_ids_k1024_uint16.npy
    └── patch_cluster_dist_k1024_fp16.npy
```

Global clustering results are stored beside the patch-token feature bank because they are reusable intermediate features.

## Local Cluster Output

```text
feature/vision/coco2014_dinov2_vitb14_reg_448_local_k20/
├── meta.json
├── images.jsonl
├── patch_tokens_shape.json
├── local_centers.pt
├── local_assignments.pt
├── local_center_weights.pt
├── resolved_config.yaml
└── patch_tokens/
    ├── shard_000_fp16.mmap
    └── shard_000.meta.json
```

`local_center_weights.pt` stores how many raw patches are assigned to each local center. Use it when interpreting local-center distributions.

## Text Atom Output

```text
feature/text/coco2014_llama_atoms_k20/
├── resolved_config.yaml
└── text_atoms.jsonl
```

Each row contains image metadata, source captions, the generated long caption, parsed atoms, and raw generation text.

## Density Feature Output

```text
feature/density/coco2014_n1000_k20/
├── resolved_config.yaml
├── sample_1000.jsonl
├── local_density_eval_original.pt
├── local_density_eval_expanded.pt
├── vision_local_centers.pt
├── text_original_exact.pt
├── text_expanded_exact.pt
├── vision_local_center_meta.jsonl
├── text_original_exact_meta.jsonl
├── text_expanded_exact_meta.jsonl
└── feature_config.json
```

These files are reusable feature banks and metadata for training or analysis.

## Density Report Output

```text
outputs/density/coco2014_n1000_k20/
├── resolved_config.yaml
└── local_density_report.json
```

Reports and metrics are experiment results, so they are stored under `outputs/`.

## Eval Prototype Output

```text
feature/eval/mscoco2017/
├── resolved_config.yaml
└── mscoco2017_eval_prototypes.pt
```

Eval prototypes are reusable evaluation features and stay under `feature/`.

## Mapping Output

```text
outputs/mapping/coco2014_n1000_k20/
├── resolved_config.yaml
├── W_<name>.pt              # mini_vec2vec
├── model_<name>.pt          # vec2vec
└── result_<name>.json
```

Exact filenames depend on `mapping.method` and experiment settings.

## Visualization Output

```text
outputs/visualize/<experiment_name>/
├── resolved_config.yaml
├── *.jpg / *.png
└── *.json / *.jsonl
```

Visualization outputs are not required by later training stages and are stored under `outputs/`.
