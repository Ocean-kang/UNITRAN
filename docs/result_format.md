# 结果格式说明

## DINOv2 feature output

```text
feature/coco2014_dinov2_vitb14_reg_448/
├── meta.json
├── images.jsonl
├── patch_tokens/
├── patch_tokens_shape.json
└── resolved_config.yaml
```

## Local cluster output

```text
feature/coco2014_dinov2_vitb14_reg_448_local_k20/
├── local_centers.pt
├── local_assignments.pt
├── local_center_weights.pt
├── image_ids.json
├── meta.json
└── resolved_config.yaml
```

`local_center_weights.pt` 表示每个 local center 覆盖的 patch 数量。做分布解释时不要忽略该权重。

## Density evaluation output

```text
outputs/local_density_n1000_k20/
├── resolved_config.yaml
├── sample_1000.jsonl
├── vision_local_centers.pt
├── text_original_exact.pt
├── text_expanded_exact.pt
└── local_density_report.json
```

其中：

- `vision_local_centers.pt`: 视觉 local centers。
- `text_original_exact.pt`: 原始 COCO captions 的 text embedding。
- `text_expanded_exact.pt`: LLM expanded text atoms 的 text embedding。
- `local_density_report.json`: 分布指标汇总。
