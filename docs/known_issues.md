# Known Issues

## 1. Llama 不应作为 density evaluation 的强依赖

推荐先离线生成 `text_atoms.jsonl`，再让 density evaluation 读取该文件。

## 2. unaligned space 指标不能直接证明语义对齐

MMD、SWD、coverage 等指标只描述几何分布差异。它们适合用来判断 text atoms 是否让语言侧密度更接近视觉侧，但不能单独证明 patch 与 atom 已经语义匹配。

## 3. Local center 必须结合权重解释

每图 local KMeans 的 center 数量固定，但每个 center 覆盖的 patch 数不同。解释视觉分布时应参考 `local_center_weights.pt`。

## 4. 不要自动模糊匹配多个候选文件

实验脚本应由 config 明确指定输入路径，避免读取错版本结果。
