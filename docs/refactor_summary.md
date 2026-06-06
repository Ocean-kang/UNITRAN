# 重构摘要

本次重构目标是把仓库从“工程库风格 + 多入口脚本混合”调整为“实验优先、配置驱动、入口统一”的研究代码仓库。

## 已完成

1. 新增 `experiments/`，作为唯一主实验入口。
2. 新增 `configs/`，每个实验对应独立 YAML。
3. 将旧 `unitran/` 迁移为 `src/`。
4. 新增 `src/*.py` flat facade，降低主流程阅读成本。
5. 移除旧 `tools/`。
6. 移除多层 `scripts/cluster|evaluate|extract|query|visualize`，改为 5 个主流程脚本。
7. 将 Llama 代码从 `third_party/llama` 移到 `external/llama`，避免被误解为 density evaluation 的默认强依赖。
8. 新增 `scripts/smoke_check.sh`。
9. 重写 README，只保留主流程命令和配置驱动说明。
10. 每个新实验入口都会保存 `resolved_config.yaml`。

## 保留的内部拆分

`src/clustering/`、`src/evaluation/`、`src/features/`、`src/visualization/` 仍保留为内部实现目录。原因是当前各功能文件较长，直接强行合并为单文件会增加 bug 风险。外部用户优先使用 `experiments/` 和 `configs/`，而不是直接调用内部模块。

## 下一步建议

1. 跑 `bash scripts/smoke_check.sh`。
2. 用 1000 张图跑通 `local_density_n1000_k20`。
3. 确认输出文件名和 README 一致。
4. 再逐步把内部长文件拆成更清晰的函数，而不是继续增加入口文件。
