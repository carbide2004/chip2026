# B 榜 E15 提交记录

## 官方反馈

- 提交时间（平台显示）：2026-10-09 10:42:06
- 提交文件：`submissions/patientphex_b_e15.jsonl`
- SHA-256：`dc2291465f39079e6e39444e514940f11de328b3cc1e0bf22de1ae34f6fd0f28`
- 成绩来源：用户提供的 B 榜平台反馈，非本地评估。

| 指标 | 得分 |
| --- | ---: |
| Score | 0.6496 |
| Task1 Mention F1 | 0.6980 |
| Task1 Document F1 | 0.7740 |
| Task2 Micro F1 | 0.6058 |
| Task2 Macro F1 | 0.5207 |

## 推理与校验

使用现有 E15 冻结方案，未依据 B 榜反馈修改阈值或 Prompt。

```bash
python scripts/run_frozen_pipeline.py \
  --input-file data/PatientPheX-V1-B/PatientPheX-V1-B.jsonl \
  --work-dir data/processed/final_b \
  --mode e15

python scripts/validate_submission.py \
  data/PatientPheX-V1-B/PatientPheX-V1-B.jsonl \
  data/processed/final_b/patientphex_e15.jsonl
```

- 文档：100
- 患者：244
- 实体提及：6,366
- 患者—表型关联：2,477
- 文件大小：838,771 字节
- 格式校验：0 错误、0 警告
- 已独立检查输入哈希、输出哈希、文档顺序、患者覆盖、实体文本偏移与 HPO 标识符。

模型权重、原始数据、推理缓存与日志位于被 Git 忽略的 `data/` 目录；提交方案及模型前提见 `b_pipeline.md`。复现需要这些本地模型与官方数据，不是仅克隆仓库即可运行。
