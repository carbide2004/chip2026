# CHIP2026 PatientPheX: 表型概念识别与患者个体化关联

- **比赛官方链接**：[CHIP2026 面向生物医学全文的特定患者表型概念识别任务 (天池)](https://tianchi.aliyun.com/competition/entrance/532512)

本项目针对 **CHIP2026 面向生物医学全文的特定患者表型概念识别任务 (PatientPheX)** 开发。
项目包含完整的**规则基线 (CPU即跑)** 与 **GPU 进阶方案 (SapBERT 软匹配 + Qwen3.5 LLM 判别/微调)**。

---

## 目录与架构设计

```text
chip2026-patientphex/
├── README.md                  # 本说明文档
├── data/
│   ├── raw/                   # 官方原始数据集 (train, A榜测试集, hp.obo)
│   └── processed/             # 预处理缓存、FAISS 向量库与 SFT 训练数据
├── src/
│   ├── data_loader.py         # HPO本体解析 (hp.obo) 与 JSONL 读写
│   ├── metrics.py             # 官方全套评测指标 (Mention F1, Doc F1, Micro F1, Macro F1)
│   ├── dictionary_extractor.py# 词典前缀索引与实体最长匹配抽取
│   ├── baseline_rule.py       # 规则基线 (CPU可直接运行，整体分 0.5675)
│   ├── sapbert_matcher.py     # [GPU] SapBERT 实体召回与 HPO 向量检索 (FAISS持久化)
│   ├── pipeline_gpu.py        # [GPU] SapBERT 软匹配端到端全量评测与提交生成
│   ├── evaluate_qwen35.py     # [GPU] 基线 vs SapBERT vs +Qwen3.5 三代方案横向对比评测
│   ├── inference_8b.py        # [GPU] Qwen3.5 + SapBERT 端到端推理脚本
│   └── prepare_sft_data.py    # [GPU] Qwen3.5 指令微调数据构造
├── scripts/
│   ├── run_optimized_v2.py    # [🌟最新推荐] 区域加权语义过滤 + 增强否定 A 榜生成脚本
│   ├── run_zeroshot_submission.py # Qwen3-8B Zero-Shot A 榜生成脚本 (得分 0.5992)
│   ├── run_fast_inference.py  # [GPU] 流式推理与 A 榜预测文件生成脚本
│   └── auto_qwen3_8b_pipeline.py # Qwen3-8B 自动化下载、Zero-shot与16-bit LoRA微调流水线
├── submissions/
│   ├── baseline_rule_A.jsonl  # 纯规则基线 A 榜预测文件 (Score: 0.5568)
│   ├── sapbert_rule_A.jsonl   # GPU SapBERT 增强版 A 榜预测文件 (Score: 0.5750)
│   ├── qwen3_8b_zeroshot_A.jsonl # Qwen3-8B Zero-Shot A 榜预测文件 (Score: 0.5992)
│   ├── qwen3_8b_v2_optimized_A.jsonl # [最新优化] V2 区域加权与全段落去噪预测文件
│   └── qwen3_8b_final_A.jsonl # SapBERT + Qwen3-8B LoRA 预测文件 (Score: 0.5161)
└── tests/
    └── test_metrics.py        # 官方指标与解析器测试
```

---

## 官方天池评测成绩（A 榜线上实测）

| 提交版本 / 方案 | 提交文件 | **总分 (Score)** | Task1 Mention F1 | Task1 Doc F1 | Task2 Micro F1 | Task2 Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **规则基线 (CPU)** | `baseline_rule_A.jsonl` | 0.5568 | 0.6087 | 0.6970 | 0.5290 | 0.3923 |
| **GPU SapBERT 软匹配** | `sapbert_rule_A.jsonl` | 0.5750 | 0.6446 *(+3.59%)* | 0.7323 *(+3.53%)* | 0.5285 | 0.3945 |
| **Qwen3-8B LoRA 微调** | `qwen3_8b_final_A.jsonl` | 0.5161 | 0.6446 | 0.7323 | 0.4116 | 0.2758 |
| **🏆 Qwen3-8B Zero-Shot (当前最优)** | `qwen3_8b_zeroshot_A.jsonl` | **0.5992** *(+4.24%)* | **0.6446** | **0.7323** | **0.5547** *(+2.57%)* | **0.4651** *(+7.28%)* |

> **关键结论与分析**：
> 1. **SapBERT 奠定 Task 1 基石**：将实体识别 Mention F1 从 0.6087 提升至 0.6446，Doc F1 从 0.6970 提升至 0.7323。
> 2. **Qwen3-8B Zero-Shot 在 Task 2 取得突破**：通过大模型上下文语义判别，精准消除了多患者家系中的假阳性串扰，**Task 2 Macro F1 从 0.3923 跃升至 0.4651（大幅提升 +7.28%）**，总分直接逼近 0.60 大关！
> 3. **LoRA 微调的教训**：小样本指令微调在开放生成时容易产生幻觉与重复 ID 导致 Precision 崩溃；未来若做微调，应采用判别式/分类式微调而非自由文本生成式。

---

## 快速上手

### 1. CPU 环境（直接运行规则基线）

无需 GPU，在当前轻量机器上即可运行：

```bash
# 1. 运行测试
python3 -m pytest tests/test_metrics.py

# 2. 运行规则基线并在训练集上评测
python3 src/baseline_rule.py
```

**当前规则基线表现 (80 篇训练集实测)**：
- **Overall Score**: `0.5675`
- Subtask 1 Mention F1: `0.6553` (P: 0.8363, R: 0.5388)
- Subtask 1 Doc F1: `0.7086` (P: 0.9148, R: 0.5782)
- Subtask 2 Micro F1: `0.4994` (P: 0.4325, R: 0.5909)
- Subtask 2 Macro F1: `0.4066` (P: 0.3845, R: 0.5630)
- **已输出 A 榜预测文件**: `submissions/baseline_rule_A.jsonl`

---

## GPU 进阶提升方案（本地显卡 / 服务器执行）

当前分析发现，训练集中 **41.88% 的金标准实体未被词典硬匹配命中**（例如缩写 `SNHL` -> `Sensorineural hearing impairment`、复数形式 `developmental delays`、同义变体 `diabetes` 等）。同时，患者关联目前仅靠距离规则，Precision 仅约 40%。

提升核心分为两步：**SapBERT 软匹配召回** 与 **8B 模型语义判别**。

### 阶段一：用 SapBERT 提升子任务 1 召回率 (Recall)

#### 1. 依赖安装
```bash
pip install torch transformers sentence-transformers faiss-gpu faiss-cpu scikit-learn
```

#### 2. 原理与流程
- **模型**：`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`（生物医学专用的实体对齐模型，仅约 400MB 显存）。
- **离线构建向量库**：
  1. 将 `hp.obo` 中 1.7 万个 HPO 概念的名字与同义词，通过 SapBERT 编码为 768 维稠密向量，存入 FAISS 索引；
  2. 对文章中未被词典硬命中的潜在医学短语（可由 Stanza 或 BioBERT 抽取），计算向量余弦相似度；
  3. 取 Top-3 候选（相似度阈值例如 > 0.82）作为有效表型。

---

### 阶段二：用 8B 模型做精准否定过滤与患者关联

#### 1. 为什么用 8B？
- **否定判断 (Subtask 1)**：消除 `ruled out` / `unremarkable` / 假阳性。
- **患者关联 (Subtask 2)**：判断一段病历中提到的症状属于哪个患者（先证者 vs 亲属 vs 既往排除病史）。

#### 2. 生成微调训练数据
在项目根目录运行数据转换脚本：
```bash
python3 src/prepare_sft_data.py
```
该脚本会将 80 篇训练集转化为 Alpaca 格式的微调数据：
- `data/processed/sft_negation.json`: 句子级表型否定二分类数据（YES / NO）
- `data/processed/sft_association.json`: 患者临床病史切片与真实确诊 HPO ID 关联问答

#### 3. 运行 8B 16-bit LoRA 微调（以 Qwen3-8B 为例）

```bash
# 执行本地自动化下载、Zero-shot评测与LoRA训练
python3 scripts/auto_qwen3_8b_pipeline.py
```

#### 4. 流式推理与提交文件生成
加载微调后的 LoRA 权重流式预测 A 榜测试集：
```bash
python3 scripts/run_fast_inference.py
```
*预测结果输出路径：`submissions/qwen3_8b_final_A.jsonl`*

---

## 本地与服务器快速同步指南 (SCP / Rsync)

你可以将服务器上的完整仓库（包含已生成的微调数据集与脚本）直接同步到本地：

```bash
# 在你本地笔记本/GPU电脑的终端运行：
scp -r ubuntu@<服务器IP>:/home/ubuntu/projects/chip2026-patientphex/ ./chip2026-patientphex/
```

同步完成后，在本地即可：
1. 直接使用现成的 SFT 数据集进行 LoRA 微调；
2. 运行 `python src/inference_8b.py` 完成 GPU 上的高质量推理预测。
