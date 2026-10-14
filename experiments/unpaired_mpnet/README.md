# UNITRAN — MSCOCO2014 Unpaired DINOv2 ↔ MPNet (Ponytail)

## 目标与约束

- **不使用 Llama-3**：直接读取 MSCOCO2014 原始 Captions，用 `sentence-transformers/all-mpnet-base-v2` 提取特征；每条 Caption L2 归一化，同图 Caption 求平均后再 L2 归一化。
- **严格无配对训练**：参照 `unpaired-rosetta/unpaired_rosetta/evaluation.py::disjoint_split`，对全部 `train2014` **Image ID** 进行一次随机排列，前后两个不重叠半区分别用于视觉/语言；可从各半区单独截取前 K 个元素。训练用不到同一 Image ID 的另一模态。
- **有配对、独立评估**：`val2014` 保留准确的 Image ID 对应关系，仅用于评估 FOSCTTM/Mean Rank/I2T 与 T2I R@K。
- **算法**：保留 UNITRAN 的 `mini_vec2vec`：KMeans+QAP → 初始 Orthogonal Procrustes → NN-ICP-style refinement → 二次 KMeans 校正。直接导入 `src/training/unitran_mapping.py` 已有的 `aligned_centroids()`、`cos_sim_matrix()`、`train_orthogonal_linear()`、`eval_cos_score()`、`set_seed()`。
- **Ponytail**：全部新代码位于 `experiments/unpaired_mpnet/`，**不覆盖** UNITRAN 的 `src/`、`configs/` 或其他实验。

## 文件及位置

将压缩包解压到 UNITRAN 仓库根目录（`/home/master/code/oymk/UNITRAN`）：

```text
UNITRAN/
├── experiments/
│   └── unpaired_mpnet/
│       ├── config.yaml          # 独立配置
│       ├── common.py            # 检查/存储公共函数
│       ├── split_coco.py        # 图像 ID 分离，保存三份 manifests
│       ├── extract_vision.py    # 使用 UNITRAN DINOv2 模块提取图像级特征
│       ├── extract_text.py      # MPNet 原始 caption 编码
│       ├── metrics.py           # 分块 topk 和真正配对验证的检索指标
│       ├── train.py             # UNITRAN 原有 mini_vec2vec 的实验专用入口
│       ├── evaluate.py          # 独立 val2014 双向检索评估
│       ├── run.sh               # 全流程顺序执行
│       ├── test_small.py        # 不需权重的小单元测试
│       └── README.md
└── INSTALL_UNITRAN_UNPAIRED_MPNET.md
```

**源码行为对照：** 原 UNITRAN 使用 448px DINOv2 Patch Tokens → 20 视觉局部中心，Llama 生成 20 Text Atoms → MPNet；原 `train_mini_vec2vec()` 还使用 `X_all[:n//2]` 与 `Y_all[n//2:]` 二次划分。本实验改为每张图像一个 DINOv2 全局向量与一条 MPNet Caption 平均向量，分组在编码之前执行。为了不改旧脚本，新 `train.py` 保持同样的训练阶段、调用旧的核心函数，去掉多余的二次划分。近邻 top-k 使用**等价的分块点积计算**避免生成整个 NxM 矩阵。

## 前置条件

你已准备好的 Conda 环境、COCO2014 可沿用。需确保 `data/mscoco14/` 下存在：

```text
data/mscoco14/
├── train2014/
├── val2014/
└── annotations/
    ├── captions_train2014.json
    └── captions_val2014.json
```

模型需要：Torch Hub 可读取 `dinov2_vitb14_reg` 权重，MPNet 可以从 HuggingFace 下载或本地缓存读取。**完全不需要 Llama-3 权重**。

```bash
conda activate /home/master/software/oymk/unitran
cd /home/master/code/oymk/UNITRAN
export CUDA_VISIBLE_DEVICES=0
python experiments/unpaired_mpnet/test_small.py
```

### 配置文件

配置 `experiments/unpaired_mpnet/config.yaml`：

```yaml
split:
  seed: 42
  train_size_per_side: 5000  # 分别选 5000 图像/5000 captions 图像，合计 10000 train2014 ID
  val_size: 1000             # 从 val2014 中抽出 1000 个真实配对用于评估
vision:
  model: dinov2_vitb14_reg
  image_size: 448
  pooling: patch_mean
text:
  model: sentence-transformers/all-mpnet-base-v2
mapping:
  seed: 0
```

`train_size_per_side: null` = 使用完整 `train2014` 的约 41,391 + 41,392 张。 `val_size: null` = 全部 40,504 张 `val2014`。修改 seed、图片数量或特征配置时**必须切换 `paths.work_dir` 和 `paths.result_dir`**，不要混用旧特征。

### 一键运行

```bash
bash experiments/unpaired_mpnet/run.sh
# 或指定自己的 YAML 配置
bash experiments/unpaired_mpnet/run.sh experiments/unpaired_mpnet/config.yaml
```

### 分步运行

```bash
# 1. 产生 ImageID 不重叠的两侧训练集合，以及独立真实配对验证集合
python experiments/unpaired_mpnet/split_coco.py

# 2. 视觉特征：只读取 Set A 训练图片 + 独立 val2014
python experiments/unpaired_mpnet/extract_vision.py --split vision_train
python experiments/unpaired_mpnet/extract_vision.py --split paired_val

# 3. 文本特征：只读取 Set B 的原始 captions + 独立 val2014 captions
python experiments/unpaired_mpnet/extract_text.py --split text_train
python experiments/unpaired_mpnet/extract_text.py --split paired_val

# 4. 保持 UNITRAN mini_vec2vec 算法训练 W
python experiments/unpaired_mpnet/train.py

# 5. val2014 的真实图文对应关系评估
python experiments/unpaired_mpnet/evaluate.py
```

## 输出文件

默认中间特征目录：

```text
feature/unpaired_mpnet/mscoco14_seed42_n10000/
├── vision_train.jsonl
├── text_train.jsonl
├── paired_val.jsonl
├── split_meta.json
├── vision_train.pt   # [5000, 768], Set A (Vision)
├── text_train.pt     # [5000, 768], Set B (Text)，Image ID 与 Set A 不相交
├── vision_val.pt     # [1000, 768], val2014
└── text_val.pt       # [1000, 768], val2014，与 vision_val.pt 逐行 ImageID 对应
```

默认结果目录：

```text
outputs/unpaired_mpnet/mscoco14_seed42_n10000/
├── W_mini_vec2vec_vision_to_text_seed0.pt
├── result_mini_vec2vec_vision_to_text_seed0.json
└── retrieval_mini_vec2vec_seed0.json
```

对 `result_*.json`，`initial_cosine`、`refine1_cosine`、`refine2_cosine` 记录原 UNITRAN 训练各阶段在验证集上的余弦分数。`retrieval_*.json` 记录更可靠的 `i2t`、`t2i` 两方向 FOSCTTM、Mean Rank、Recall@1/5/10，实际根据 val2014 同 Image ID 进行真值匹配。

## 与原版算法一致性和差异

| 内容 | 本方案 | 原 UNITRAN mini-vec2vec |
|---|---|---|
| KMeans/QAP、anchor runs | 直接复用 `aligned_centroids()` | 相同 |
| Procrustes 初始化 | 直接复用 `train_orthogonal_linear()` | 相同 |
| 迭代求近邻、更新 W | 相同公式和超参数；top-k 相似度分块计算 | 整矩阵计算 |
| 末尾聚类校正 | 相同 KMeans/Procrustes/EMA | 相同 |
| 数据划分 | 事先按 **Image ID** 无交集划分 | 训练函数中按特征行取前后半 |
| 输入粒度 | 每图一个全局视觉向量 + MPNet Caption 均值 | 每图 K=20 个局部视觉/文本特征 |
| Llama | 不使用 | Expanded Text 方案使用 |
| 验证真值 | `val2014` 同 Image ID | 原始默认按照特征行假定真值 |

> 注意：`mini_vec2vec` 在这里没有新的可训练神经网络模型，训练输出是映射矩阵 `W`。`training.epochs` 仅适用于 UNITRAN 的 `vec2vec` 分支，本实验不使用。

## 计算规模与注意事项

- 首次建议按默认 5000 / 5000 + 1000 验证运行；想先做最小 Smoke Test，可以把 `train_size_per_side` 改为 `1000`、`val_size` 改为 `100`，同时改新输出路径。训练默认 `refine2_clusters=500`，每边至少需 500 张图像。
- 全量 COCO2014 约 41k/41k，原 mini-vec2vec KMeans/QAP 和 `refine_iters=100` 的耗时将显著增长；已经将 top-k 和验证 retrieval 分块化，仍需满足 CPU RAM 和耗时条件。
- 若模型权重只能离线访问：DINOv2 读取 `DINOV2_REPO` 或 Torch Hub cache；MPNet `text.model` 可以填写模型本地目录。
- 如更接近 unpaired-rosetta 的视觉侧，改为 `vision.model: dinov2_vitb14`、`vision.image_size: 224`、`vision.pooling: all_token_mean` 并使用新的 `paths.work_dir`。不要与 `dinov2_vitb14_reg` 的结果直接混作同一 baseline。
- Validation 是带有真实对应信息的数据集，但仅供评估；训练没有访问 Val Pair ID 关系。将 `val_size=null` 后评估 40504 对时会很慢，`evaluate.py` 的分块检索会避免一次性物化 40504² 矩阵。

## 验证正确性

1. `split_meta.json` 显示 A/B 无交集且两个集合数量正确；训练时 `common.assert_disjoint_and_paired()` 再次验证。
2. `*.pt` 中每个特征对应一个 Image ID，向量为 768 维，并且不存在 NaN/Inf。
3. `retrieval_*.json` 的 `num_validation_pairs` 与 `val_size` 一致；指标范围合理：`foscttm ∈ [0,1]`、`mean_rank ∈ [1,N]`。
4. 只更改特征方案和无配对构建，没有修改主仓库任何原有 Python 文件。
