# UNITRAN

UNITRAN 是一个轻量级的 text / vision embedding 空间无配对翻译项目。

当前仓库包含 3 个可运行脚本：

- `main.py`：在不使用成对训练样本的情况下，学习 vision embedding 到 text embedding 的正交映射 `W`。
- `distribution_metric.py`：计算两个 embedding 集合之间的分布差距指标。
- `visualize_embeddings_2d.py`：使用 PCA 或 t-SNE 将 text / vision embedding 可视化到二维空间。

## 代码逻辑

### `main.py`

`main.py` 需要 `--embedding_dir` 目录下存在两个文件：

```text
talk2dino_avg_self_attn_out_80000.pt
val_paired_8192.pt
```

每个文件应包含以下两个 key：

```text
vision_feats
text_feats
```

整体流程如下：

1. 加载 train 和 val 特征。
2. 如果 vision/text 特征维度不同，使用 zero padding 补到相同维度。
3. 对每个模态分别做中心化和 L2 normalize。
4. 使用无配对训练切分：`X_train = vision split 1`，`Y_train = text split 2`。
5. 两个模态分别做 KMeans，然后用 quadratic assignment 对齐两个聚类中心的结构。
6. 根据“样本到聚类中心的相似度签名”构造初始伪匹配。
7. 用 Orthogonal Procrustes 拟合正交映射 `W`。
8. 通过 nearest-neighbor ICP-style refinement 和最终 cluster-based correction 继续更新 `W`。
9. 在 paired validation 文件上评估，并保存：

```text
outputs/W_<source>_to_<target>_seed<seed>.pt
outputs/result_<source>_to_<target>_seed<seed>.json
```

### `distribution_metric.py`

这个脚本可以加载一个同时包含两个模态的 `.pt` 文件，也可以加载两个独立 `.pt` 文件。支持 3 种预处理方式：

- `raw`：只做 zero padding。
- `l2`：zero padding 后做 L2 normalize。
- `unitran`：zero padding 后，对每个模态中心化，再做 L2 normalize。

当前计算的指标包括：

- all-pair cosine similarity mean
- same-index cosine similarity mean，只有等长且同索引可视为 paired 时才计算
- RBF MMD
- KMeans-center Wasserstein distance，使用 Hungarian matching 匹配聚类中心
- precision / recall coverage ratio
- pairwise Spearman structure consistency
- entropic GW distance

### `visualize_embeddings_2d.py`

这个脚本加载 text/vision embedding，从两个模态采样相同数量的点，使用 PCA 或 t-SNE 投影到二维空间；如果传入已学习的 `W`，还会额外画出应用 `W` 后的对齐效果。

## 推荐环境

4090D 服务器已经有 CUDA 12.8 driver。本项目只需要 PyTorch 和少量科学计算依赖，推荐使用：

- Python 3.10
- PyTorch CUDA 12.8 wheel
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- tqdm

## 安装环境

在仓库根目录执行：

```bash
conda env create -f conda.yaml
conda activate unitran
```

检查环境是否可用：

```bash
python - <<'PY'
import torch
import numpy as np
import scipy
import sklearn
import matplotlib
import tqdm

print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda runtime:', torch.version.cuda)
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

检查项目脚本是否有语法错误：

```bash
python -m py_compile main.py distribution_metric.py visualize_embeddings_2d.py
```

## 运行示例

### 训练 UNITRAN 映射

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --embedding_dir /path/to/embedding_dir \
  --source dinov2 \
  --target text \
  --out_dir outputs/unitran \
  --seed 0
```

### 计算分布指标

如果一个 `.pt` 文件同时包含 `vision_feats` 和 `text_feats`：

```bash
python distribution_metric.py \
  --input_pt /path/to/features.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --preprocess unitran \
  --max_points 2000 \
  --out_json outputs/distribution_metric.json
```

如果 vision/text 是两个独立 `.pt` 文件：

```bash
python distribution_metric.py \
  --vision_pt /path/to/vision.pt \
  --text_pt /path/to/text.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --preprocess unitran \
  --out_json outputs/distribution_metric.json
```

### 可视化 embedding

```bash
python visualize_embeddings_2d.py \
  --input_pt /path/to/features.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --method pca \
  --preprocess unitran \
  --out_dir outputs/vis
```

可视化应用 `W` 前后的结果：

```bash
python visualize_embeddings_2d.py \
  --input_pt /path/to/features.pt \
  --vision_key vision_feats \
  --text_key text_feats \
  --W outputs/unitran/W_dinov2_to_text_seed0.pt \
  --method pca \
  --preprocess unitran \
  --out_dir outputs/vis
```

## 注意事项

- 这次只新增环境文件和 README 安装说明，没有改动算法代码。
- `main.py` 当前在 `--embedding_dir` 下硬编码读取两个文件名：`talk2dino_avg_self_attn_out_80000.pt` 和 `val_paired_8192.pt`。
- 大规模 cosine similarity 矩阵会占用较多 CPU/GPU 内存。如果显存或内存紧张，优先调小 `--refine_sample`、`--num_anchor_runs`、`--anchor_subsample`，或者调小 metric 脚本里的采样点数。
