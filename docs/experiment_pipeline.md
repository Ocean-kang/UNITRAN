# UNITRAN 实验流程

## 目标

该流程用于比较 DINOv2 视觉 patch/local-center 特征与 text atoms 特征的分布差异，并为后续 unpaired mapping 训练准备输入。

## 流程

1. 提取 DINOv2 patch tokens。
2. 对每张图的 patch tokens 做 local KMeans。
3. 从 COCO captions / LLM recaption 构造 text atoms。
4. 使用 sentence-transformer 提取 text embeddings。
5. 对比 original captions、expanded text atoms 与 vision local centers 的密度差异。
6. 使用可视化检查 cluster / local center 是否有明确视觉语义。
7. 使用 vision/text 特征训练 UNITRAN / vec2vec-style mapping。

## 入口

统一入口格式：

```bash
python experiments/xxx.py --config configs/xxx.yaml
```

不要同时维护 shell 默认参数、README 默认参数、Python 默认参数三套配置。
