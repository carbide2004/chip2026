# PatientPheX B榜冻结推理

## 前提

- 将官方B榜JSONL保存到独立路径，不修改原文件。
- 保留以下冻结模型：
  - `data/processed/final/token_ner/model/`
  - `data/processed/final/marker_e15_model/`
  - `data/models/Qwen3-8B-hf/`
- E15阈值固定为Marker 0.60、SapBERT 0.90、关系0.65、Qwen区间0.35–0.85。

## 一键运行

```bash
python scripts/run_frozen_pipeline.py \
  --input-file /path/to/PatientPheX-B.jsonl \
  --work-dir data/processed/final_b \
  --mode both
```

如果已有对应数据集的Qwen缓存，可加：

```bash
  --e15-seed-cache /path/to/e15_cache.jsonl \
  --e7-seed-cache /path/to/e7_cache.jsonl
```

缓存只能复用相同PMC、patient ID和HPO key；脚本会为缺失key继续推理。

## 输出

```text
data/processed/final_b/
  patientphex_e15.jsonl
  patientphex_e7.jsonl
  e15_entities.jsonl
  e7_entities.jsonl
  e15_qwen_cache.jsonl
  e7_qwen_cache.jsonl
  e15_summary.json
  e7_summary.json
  manifest.json
```

`manifest.json`记录输入、训练集、本体、BIO模型、Marker模型和提交文件SHA256。

## 提交前检查

```bash
python scripts/validate_submission.py \
  /path/to/PatientPheX-B.jsonl \
  data/processed/final_b/patientphex_e15.jsonl
sha256sum data/processed/final_b/patientphex_e15.jsonl
```

主提交使用E15；E7仅作为回退。不要在看到B榜反馈后修改冻结阈值或Prompt。
