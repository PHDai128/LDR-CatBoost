# LDR-CatBoost 论文代码

本目录只保留 V2_1006 论文定量实验所需的代码。三座高炉在公开代码中统一记为 `furnace_1`、`furnace_2`、`furnace_3`。主实验、消融、配对统计和延迟敏感性的对应入口见 [README.md](README.md) 中的表格。

运行前，把**有权使用的**三份小时数据在本地分别命名为 `data/furnace_1.csv`、`data/furnace_2.csv`、`data/furnace_3.csv`。`data/` 和 `results/` 中的数据、逐样本预测及模型文件由 `.gitignore` 排除，不能随公开仓库上传。原始数据和现有实验目录没有被修改。

依次运行主实验、配对 bootstrap、受控消融、延迟敏感性，再生成支持表与图。具体命令、输入格式和论文表图映射见英文 README。主实验的历史内部名称 `Standard CDR-CatBoost` 对应论文中的 LDR-CatBoost；受控消融应使用**同一批次重跑的完整模型**作对照，不要和主实验完整模型的数值混算。

合成数据检查命令：

```bash
python -m unittest discover -s tests -v
```

已用本地数据对公开版的延迟敏感性入口完成炉 1、2 小时延迟的三折试跑：测试样本数和 HR@0.05 与原始记录一致，HR@0.10 和 MAE 略有差异。论文数值以原始实验记录为准；跨 GPU 环境运行不保证数值完全相同，具体数值见英文 README。

本包尚未上传 GitHub。公开前仍需由作者确认代码发布权限，并决定是否添加许可证。
