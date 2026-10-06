# LDR-CatBoost 论文代码

本仓库包含论文 **“LDR-CatBoost: A Laboratory-Delay-Aware, Persistence-Anchored Residual Framework for Next-Hour Hot-Metal Silicon Prediction”** 实验使用的代码，仅保留主要模型、对比模型、受控消融、配对统计和发布延迟敏感性实验。仓库不包含画图代码或工业数据。

| 实验 | 运行入口 |
| --- | --- |
| LDR-CatBoost 与 Persistence、XGBoost、DLinear、iTransformer、PatchTST 对比 | `src/run_final_experiments.py` |
| 特征、预测目标与损失函数的受控消融 | `analysis/controlled_component_study_20261005.py` |
| 配对 block-bootstrap 统计比较 | `analysis/paired_baseline_bootstrap_20261005.py` |
| 0、1、2、4 小时发布延迟敏感性 | `src/run_delay_sensitivity.py` |

`src/` 中的其他文件是这些实验所需的数据预处理、特征构建、时间划分和模型定义。具体运行命令见 [英文 README](README.md)。运行前，按 [数据说明](data/README.md) 将有权使用的三座高炉小时数据放在本地 `data/` 目录。数据、逐样本预测和训练模型不会随代码提交。

代码对已有小时 Si 序列应用 `shift(delay).ffill()`，以限定预测时可使用的 Si 信息。实验使用假设的固定发布延迟；仓库不包含小时 Si 标签的上游生成代码，也不还原真实化验报告发布时间。不同 GPU 或软件版本的重跑指标可能略有差异，论文数值以原始实验记录为准。
