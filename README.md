# UNITRAN

UNITRAN 是一个面向研究迭代的实验代码仓库，当前目标是研究 DINOv2 patch/local-center 视觉特征与 LLM 生成的 text atoms 在分布层面的差异，并为后续 UNITRAN / vec2vec-style unpaired mapping 训练提供可复现输入。

本仓库不追求通用 Python package 风格，优先保证：

1. 实验入口少。
2. 每个实验有固定 config。
3. 输入、输出、命令清楚。
4. 结果可复现、可检查。
5. 后续 idea 容易快速插入。

## 1. 当前仓库结构

```text
UNITRAN/
├── README.md
├── conda.yaml
├── configs/                         # 每个实验的 YAML 配置
│   ├── extract_dinov2.yaml
│   ├── global_cluster_k1024.yaml
│   ├── local_cluster_k20.yaml
│   ├── text_atoms_llama.yaml
│   ├── local_density_n1000_k20.yaml
│   ├── local_density_n5000_k20.yaml
│   ├── visualize_cluster.yaml
│   └── train_mapping.yaml
├── experiments/                     # 统一实验入口
│   ├── extract_dinov2_features.py
│   ├── run_global_patch_cluster.py
│   ├── run_local_patch_cluster.py
│   ├── run_text_atoms.py
│   ├── run_density_eval.py
│   ├── run_visualize_cluster.py
│   └── train_mapping.py
├── scripts/                         # 仅保留少量快捷脚本
│   ├── run_1_extract.sh
│   ├── run_2_global_cluster.sh
│   ├── run_2_local_cluster.sh
│   ├── run_3_text_atoms.sh
│   ├── run_4_density_eval.sh
│   ├── run_5_visualize.sh
│   └── smoke_check.sh
├── src/                             # 可复用实现与轻量 facade
│   ├── io.py
│   ├── dino.py
│   ├── cluster.py
│   ├── text_atoms.py
│   ├── metrics.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── mapping.py
│   ├── experiment_utils.py
│   ├── clustering/
│   ├── datasets/
│   ├── embeddings/
│   ├── evaluation/
│   ├── features/
│   ├── training/
│   ├── utils/
│   └── visualization/
├── external/llama/                  # Llama 相关代码，不作为 density eval 强依赖
├── outputs/
├── data/
└── docs/
```

核心变化：旧的 `tools/` 和多层 `scripts/*/*.sh` 已移除；主流程统一改为：

```bash
python experiments/xxx.py --config configs/xxx.yaml
```

## 2. 环境

```bash
conda env create -f conda.yaml
conda activate unitran
```

如果只做 smoke check，不需要真实 COCO 数据或 GPU；如果要实际跑 DINOv2 / text embedding / Llama，需要相应模型、数据和 CUDA 环境。

## 3. 主流程命令

### Step 0: smoke check

每次重构或迁移后先跑：

```bash
bash scripts/smoke_check.sh
```

它会检查：

```bash
python -m compileall -q src experiments
python experiments/extract_dinov2_features.py --help
python experiments/run_global_patch_cluster.py --help
python experiments/run_local_patch_cluster.py --help
python experiments/run_density_eval.py --help
python experiments/run_visualize_cluster.py --help
python experiments/train_mapping.py --help
```

### Step 1: 提取 DINOv2 patch tokens

配置文件：`configs/extract_dinov2.yaml`

```bash
python experiments/extract_dinov2_features.py --config configs/extract_dinov2.yaml
# 或
bash scripts/run_1_extract.sh
```

主要输入：

```text
data/mscoco14/train2014 或 config 中指定的 image_dir
```

主要输出：

```text
feature/coco2014_dinov2_vitb14_448/
├── meta.json
├── images.jsonl
├── patch_tokens/
├── patch_tokens_shape.json
└── resolved_config.yaml
```

### Step 2A: 所有 patch tokens 全局 clustering

这一步是不做每图 local 压缩的对照组：直接把所有图片的所有 DINOv2 patch tokens 输入 streaming KMeans。

配置文件：`configs/global_cluster_k1024.yaml`

```bash
python experiments/run_global_patch_cluster.py --config configs/global_cluster_k1024.yaml
# 或
bash scripts/run_2_global_cluster.sh
```

主要输入：

```text
feature/coco2014_dinov2_vitb14_448/patch_tokens/
```

主要输出：

```text
feature/coco2014_dinov2_vitb14_448/
├── kmeans/
│   ├── centroids_k1024_fp32.npy
│   ├── centroids_k1024_fp16.npy
│   ├── cluster_counts_k1024.npy
│   └── train_log_k1024.jsonl
├── assignment/
│   ├── patch_cluster_ids_k1024_uint16.npy
│   └── patch_cluster_dist_k1024_fp16.npy
└── resolved_config.yaml
```

### Step 2B: 每图 local patch clustering

这一步是 local 压缩实验组：每张图内部先聚成固定数量的 local centers，再和 text atoms 做密度比较。

配置文件：`configs/local_cluster_k20.yaml`

```bash
python experiments/run_local_patch_cluster.py --config configs/local_cluster_k20.yaml
# 或
bash scripts/run_2_local_cluster.sh
```

主要输入：

```text
feature/coco2014_dinov2_vitb14_448/patch_tokens/
```

主要输出：

```text
feature/coco2014_dinov2_vitb14_448_local_k20/
├── local_centers.pt
├── local_assignments.pt
├── local_center_weights.pt
├── image_ids.json
├── meta.json
└── resolved_config.yaml
```

注意：`local_center_weights.pt` 用于记录每个 local center 覆盖的 patch 数量，避免把所有 center 错误理解为等权视觉分布。

### Step 3: 构造 text atoms

配置文件：`configs/text_atoms_llama.yaml`

```bash
python experiments/run_text_atoms.py --config configs/text_atoms_llama.yaml
# 或
bash scripts/run_3_text_atoms.sh
```

主要输入：

```text
outputs/local_density_n1000_k20/sample_1000.jsonl
```

主要输出：

```text
outputs/text_atoms_llama_k20/text_atoms.jsonl
```

说明：Llama 只负责提前生成 text atoms；density evaluation 不默认强依赖 Llama，只读取已经存在的 `recaption_jsonl`。

### Step 4: density evaluation

配置文件：`configs/local_density_n1000_k20.yaml`

```bash
python experiments/run_density_eval.py --config configs/local_density_n1000_k20.yaml --stage all
# 或
bash scripts/run_4_density_eval.sh
```

`--stage` 可选：

```text
sample    只生成实验样本 jsonl
features  只提取 original / expanded text embeddings 并配平 vision local centers
report    只计算 metrics
all       依次执行 sample -> features -> report
```

主要输出：

```text
outputs/local_density_n1000_k20/
├── resolved_config.yaml
├── sample_1000.jsonl
├── vision_local_centers.pt
├── text_original_exact.pt
├── text_expanded_exact.pt
└── local_density_report.json
```

### Step 5: 可视化

配置文件：`configs/visualize_cluster.yaml`

```bash
python experiments/run_visualize_cluster.py --config configs/visualize_cluster.yaml
# 或
bash scripts/run_5_visualize.sh
```

通过 `visualization.mode` 切换可视化类型：

```yaml
visualization:
  mode: cluster_topk
```

可选值：

```text
cluster_topk
cluster_extremes
image_cls_attention
image_local_centers
embedding_2d
```

`cluster_extremes` 用于查看每个全局 cluster center 的 most/least assigned patches。输出中每组最多保留一张图一个 patch，避免 top-k 被同一张图重复占满；主图为 `image+crop` 拼图，attention map 另存为独立拼图，并额外写出 `cluster_stats.jsonl` 记录 cluster size、similarity min/mean/max 和 distance min/mean/max。

### Step 6: mapping training

配置文件：`configs/train_mapping.yaml`

```bash
python experiments/train_mapping.py --config configs/train_mapping.yaml
```

`mapping.method` 用于切换训练方法：

```yaml
mapping:
  method: mini_vec2vec   # 或 vec2vec
```

当前同一个入口支持两种方法：`mini_vec2vec` 保留原来的 centroid/QAP + Procrustes + ICP-style refinement；`vec2vec` 使用 adversarial + reconstruction + cycle-consistency + VSP 的无配对映射训练。新增方法时只需要在 `src/training/unitran_mapping.py` 的 `METHODS` 注册表中增加一个训练函数。

## 4. 推荐的小样本实验顺序

当前建议优先跑通 1000 张图、每图 20 个 local center。若要做 local 与 non-local 对照，在特征提取后同时跑 Step 2A 和 Step 2B：

```bash
bash scripts/smoke_check.sh
bash scripts/run_1_extract.sh
bash scripts/run_2_global_cluster.sh
bash scripts/run_2_local_cluster.sh
python experiments/run_density_eval.py --config configs/local_density_n1000_k20.yaml --stage sample
python experiments/run_text_atoms.py --config configs/text_atoms_llama.yaml
python experiments/run_density_eval.py --config configs/local_density_n1000_k20.yaml --stage features
python experiments/run_density_eval.py --config configs/local_density_n1000_k20.yaml --stage report
```

如果已经有 `text_atoms.jsonl`，可以跳过 Llama 生成步骤，直接在 `configs/local_density_n1000_k20.yaml` 中修改：

```yaml
text:
  recaption_jsonl: path/to/your/text_atoms.jsonl
```

## 5. 配置规则

每个实验脚本只从 config 读取默认参数。需要修改路径、数量、K 值、模型、输出目录时，优先改 YAML，不要改 Python 代码。

每次运行会在输出目录保存：

```text
resolved_config.yaml
```

这样可以回溯某次实验真实使用的参数。

## 6. 结果解释重点

当前 density evaluation 主要比较：

1. original COCO captions embedding 与 raw patch tokens / vision local centers 的分布差异。
2. expanded text atoms embedding 与 raw patch tokens / vision local centers 的分布差异。
3. expanded 是否在 MMD、sliced Wasserstein、coverage、precision/recall 等指标上更接近 vision 分布。

需要特别注意：

1. global clustering 是所有 patch tokens 的数据集级聚类，用于 non-local 对照和全局 cluster 可视化。
2. local center 是每图内部压缩后的视觉 token 表示，不等价于全局 object category。
3. text atoms 的数量建议与 local_k 对齐，比如每图 20 个 local centers，对应每图 20 个 text atoms。
4. local center 权重很重要，因为有些 center 可能只覆盖极少 patch，有些 center 覆盖大面积背景或主体区域。
5. unaligned space 的 density 指标只能说明几何/密度差异，不能直接证明语义对齐成功。

## 7. 不再维护的旧入口

旧版本中的这些入口已移除或不再作为主流程维护：

```text
tools/*.py
scripts/cluster/*.sh
scripts/evaluate/*.sh
scripts/extract/*.sh
scripts/query/*.sh
scripts/visualize/*.sh
```

请使用：

```bash
python experiments/xxx.py --config configs/xxx.yaml
```

## 8. 文档

更多说明见：

```text
docs/experiment_pipeline.md
docs/result_format.md
docs/known_issues.md
docs/refactor_summary.md
```
