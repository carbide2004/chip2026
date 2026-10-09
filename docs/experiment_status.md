# PatientPheX 实验状态

更新时间：当前工作区。原始数据、模型和生成产物位于被Git忽略的 `data/` 或用户缓存中。

## 评测边界

- 所有已报告的新实验只使用64篇development pool的4折；16篇locked validation尚未打开。
- association分号拆分仍是待官方确认的临时口径。
- A榜线上结果尚未用于选择新方案。
- E3 fold0的0.90阈值是在fold0选择的；随后固定到fold1复核。
- E4的0.65关系阈值由四折均值选择，因此E5不是该阈值的独立测试；最终需在locked validation确认一次。

## E0：数据和评测基础设施

产物：`data/processed/e0/`

- 数据清单及SHA256；
- 125项原始标注问题记录；
- 64/16开发池与锁定验证划分；
- 4个16篇CV validation fold；
- 未发现Jaccard>=0.8的训练近重复文档组；
- 本地评测修复：pred=-1不再通配有效gold HPO，gold=-1仍按span匹配，且匹配一对一；
- 提交validator已实现；
- 当前15项自动测试通过。

## E1：可复现词典与位置基线

四折均值最佳：HPO最长词典匹配、不启用简单否定、患者与实体同passage关联。

| 指标 | 均值 |
|---|---:|
| 总分 | 0.6102 |
| Mention F1 | 0.6635 |
| Document F1 | 0.7167 |
| Patient Micro F1 | 0.5682 |
| Patient Macro F1 | 0.4925 |

简单否定规则和输出所有重叠词典命中均下降，因此未进入默认配置。

## E2：gold span条件下的HPO候选检索

数据单位：64篇开发池内4754个可解析、指定分支内的阳性mention-ID单元。

| 方法 | R@1 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|
| 字符ratio | 0.7408 | 0.8422 | 0.8721 | 0.8902 |
| token-set | 0.6891 | 0.8563 | 0.8885 | 0.9129 |
| SapBERT dense | **0.8643** | **0.9478** | **0.9634** | **0.9756** |
| dense+lexical RRF | 0.7928 | 0.9285 | 0.9571 | 0.9750 |

SapBERT资源：`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`，revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`，109,482,240参数。RRF不优于dense，默认不采用。

## E3：学习型实体边界

模型：`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`，revision `e1354b7a3a09615f6aba48dfad4b7a613eef7062`。首个BIO基线不能完整表达嵌套实体，训练脚本显式报告被投影丢弃的重叠span。

- fold0：未校准span F1 0.6453；阈值0.90后0.6750；与词典简单融合0.6992。
- fold1固定0.90复核：token span F1 0.6643；融合0.6908。
- 端到端融合不稳定：fold0优于词典，fold1低于词典。因此学习模型只保留为候选召回来源，尚未替换E1默认实体系统。

## E4/E5：患者—概念证据集合

用患者-概念bag作为训练单位，不伪造mention级患者标签。特征仅使用距离、same passage、最近患者、出现次数、方向和section等可审计证据。

Gold实体条件，固定逻辑回归阈值0.65四折平均：

| 方法 | Micro F1 | Macro F1 |
|---|---:|---:|
| Same passage | 0.7672 | 0.6984 |
| 证据集合逻辑回归 | **0.8026** | **0.7522** |

接入词典预测实体后的四折均值：

| 方法 | 总分 | Micro F1 | Macro F1 |
|---|---:|---:|---:|
| Same passage | 0.6102 | 0.5682 | 0.4925 |
| Gold候选训练的证据集合模型 | **0.6193** | **0.5824** | **0.5145** |
| 预测候选训练 | 0.6161 | 0.5762 | 0.5080 |

四个fold中证据集合模型均优于same-passage。当前最佳可复现开发方案为：最长词典实体 + 证据集合逻辑回归。

## E6：Qwen3-8B不确定性路由

模型：`Qwen/Qwen3-8B`，官方revision `b968826d9c46dd6066d109eabc6255188de91218`，8,190,735,360参数，Apache-2.0。本地五个权重分片的字节大小与该revision一致并通过Safetensors结构检查。

Qwen只处理逻辑回归概率0.35–0.85的患者-概念候选，输出严格限制为YES/NO/UNCERTAIN；其余候选保留轻量模型判断。四折均使用同一Prompt和区间，逐例结果已缓存。

| 方法 | 总分 | Micro F1 | Macro F1 |
|---|---:|---:|---:|
| 逻辑回归，统一0.65 | 0.6193 | 0.5824 | 0.5145 |
| 逻辑回归，单/多患者0.35/0.60 | 0.6218 | 0.5892 | 0.5179 |
| Qwen不确定性路由 | **0.6296** | **0.6050** | **0.5331** |

四个fold总分分别提升约0.0208、0.0048、0.0116、0.0042。分层阈值与统一阈值在当前Qwen路由区间下产生相同最终输出。该实验仍属于开发池CV，不是线上成绩。

## E7：完整集成与一次性保留集

完整链路为：最长HPO词典 + BiomedBERT实体候选（概率>=0.90）+ SapBERT门控（余弦>=0.90）+ 患者-概念证据集合逻辑回归 + Qwen3-8B不确定性路由。

四折平均：

| 指标 | 结果 |
|---|---:|
| Mention F1 | 0.7138 |
| Document F1 | 0.7873 |
| Patient Micro F1 | 0.6546 |
| Patient Macro F1 | 0.5787 |
| 总分 | **0.6836** |

架构冻结后，用64篇开发池重新训练，在16篇保留集只评估一次，未按结果调参：

| 指标 | 结果 |
|---|---:|
| Mention F1 | 0.7025 |
| Document F1 | 0.7595 |
| Patient Micro F1 | 0.6488 |
| Patient Macro F1 | 0.5848 |
| 总分 | **0.6739** |

CV到保留集下降约0.0097。保留集中Qwen路由将总分从0.6540提高到0.6739。

## A榜冻结提交

使用全部80篇训练，固定5轮、NER阈值0.90、SapBERT门槛0.90、关系阈值0.65、Qwen路由区间0.35-0.85，生成：

`submissions/patientphex_a_frozen_e7.jsonl`

- 20篇文档、53位患者、1341个实体、523个患者关联；
- Qwen处理267个不确定候选；
- 提交validator：0 error、0 warning；
- SHA256：`0b16cb15f639b7654dcab6e1e180e71d81d3d9f0cf4b97c1f2ceac1d9af3dba4`。

E7线上A榜：总分0.6959，Mention 0.7027，Document 0.7729，Micro 0.6508，Macro 0.6572。

## E8-E12后续实验

已否决：增加患者锚点的Qwen Prompt（过度预测YES）、扩大Qwen路由到0.10-0.95、Prompt投票/元分类器、训练语料表面词典扫描、GlobalPointer跨度模型。它们均未稳定超过E7。

E12使用BiomedBERT直接编码患者别名、其他患者、HPO及多处上下文，不涉及LoRA。单模型随机种子波动明显，因此冻结为两个种子概率平均，并仅作为E7阳性结果的保守复核：`E7=YES AND mean(E12)>=0.20`。

| 方法 | 四折总分 | Micro | Macro |
|---|---:|---:|---:|
| E7 | 0.6836 | 0.6546 | 0.5787 |
| E7与双种子E12交集 | **0.6923** | **0.6576** | **0.6107** |

四折分别提升0.0084、0.0056、0.0087、0.0123。该规则未再次查看已打开的locked validation。第二版候选：

`submissions/patientphex_a_e12_intersection.jsonl`

- 保留E7全部1341个实体；患者关联由523降至459；
- validator：0 error、0 warning；
- SHA256：`28cd1458083d874ba498c38fafef2b66ef45fa762e0017693be9a51cfba05f99`。

E12交集线上A榜：总分0.6876，Mention 0.7027，Document 0.7729，Micro 0.6443，Macro 0.6304。相较E7总分下降0.0083，Micro下降0.0065，Macro下降0.0268；任务1完全相同。E12从523项关联删除64项，在A榜上删掉了过多真阳性。该方案正式否决，不再通过A榜反馈调其阈值。

## E13：论文方法适配实验

根据ATLOP/DREEAM/EIDER，将patient-HPO bag拆为最多5个occurrence实例，使用BiomedBERT注意力MIL，并融合全文均值、潜在证据和最大实例分数。该模型不需要mention级患者标签，但fold0仅得0.6241（Micro 0.5572、Macro 0.4490），主要问题是召回降至0.4883。只把其高置信结果用于补充E7时，fold0最大增益约0.0049，低于上线门槛，停止扩展。

Maverick启发的轻量患者cluster-seed特征使用官方patient mentions构造局部唯一患者、前序/后序患者、同passage所有者和窗口所有者。结合已有Qwen缓存后四折0.6854，相对E7仅提升0.0018，且fold2/3略降，因此不生成线上提交。

## E14/E15：PL-Marker式候选span分类

候选由HPO词典与BIO阈值0.30并集产生；fold0候选边界召回0.7997。BiomedBERT使用`[unused1]/[unused2]`标记候选边界做二分类，词典实体无条件保留，新增span再经SapBERT 0.90门控。marker阈值0.60和SapBERT阈值0.90只在fold0联合选择，fold1-3固定。

| 方法 | 总分 | Mention | Document | Micro | Macro |
|---|---:|---:|---:|---:|---:|
| E7 | 0.6836 | 0.7138 | 0.7873 | 0.6546 | 0.5787 |
| E15 Marker + E7关系 | **0.6905** | **0.7221** | **0.7914** | **0.6588** | **0.5896** |

四折总分分别提升0.0094、0.0051、0.0083、0.0046；第二随机种子在fold0也保持任务1提升。全80篇训练后A榜候选：

`submissions/patientphex_a_e15_marker.jsonl`

- 1328个实体、520项患者关联、Qwen路由274项；
- validator：0 error、0 warning；
- SHA256：`3b3f8f65a66eb087f77528f80175f7cd3a8ac67db648d707de616b2ee6f59bea`。

E15线上A榜：总分**0.7010**，Mention 0.7045，Document 0.7758，Micro 0.6545，Macro 0.6692。相较E7四项全部提高：总分+0.0051、Mention +0.0018、Document +0.0029、Micro +0.0037、Macro +0.0120。E15成为当前最佳，且线上增益方向与四折一致。

## E16-E18：患者归属后续实验

E15实体候选对gold患者关联的概念覆盖率约0.770；归属完全正确时Micro/Macro理论上限约0.869/0.862，说明主要剩余瓶颈是患者归属。

- E16使用EIDER式银证据：由bag标签、同passage唯一患者和最近患者构造occurrence标签，再以max聚合。fold0总分仅0.5601、Macro 0.2132，银规则偏向容易样本并导致大量患者空预测，否决。
- 同文同HPO患者listwise排序：保持E15预测数量，只做Top-K重分配。四折0.6867，低于E15的0.6905，否决。
- E17让Qwen一次为单个occurrence选择患者ID/BACKGROUND/NONE/UNCERTAIN。最安全的“仅在E15无预测时补充”fold0为0.6799，低于E15 fold0的0.6832，否决。
- E18从训练fold检索最相似正负示例做few-shot Qwen路由。fold0为0.6765，且YES偏置加重，否决。

## 患者叙事分段与别名扩展

比较显式同段、同章节前向继承、有限距离前向继承、最近患者段落和同章节最近段落。只有显式同段稳定；所有传播规则平均下降，说明医学文章在病例、家系和讨论之间频繁切换，不能按段落顺序继承主体。

正文patient/case/proband/亲属模式共1865次，官方mentions仅覆盖285次。保守扩展编号、谱系编号、唯一亲属角色、明确proband和单患者泛指后新增约1038个mention；直接作为anchor四折降至0.6855。保留官方视图并把扩展别名仅作为附加特征时为0.6904，与E15 0.6905等价，无增益。该路线停止。

进一步错误分层：单患者Micro约0.700，2-3患者最低约0.640（Precision 0.612），4+患者约0.652；按患者数约束Qwen YES未超过E15。BIO边界±词变体不增加gold候选命中。忽略标点/连字符的柔性HPO词典将fold0候选召回从0.7997提高到0.8107，但经Marker与SapBERT后Mention 0.7250、Document 0.7826，综合低于E15 fold0，停止扩展。

## E20-E21：动态潜变量归属

实现CLAG潜变量图：每个HPO occurrence对各患者产生软归属概率，通过Noisy-OR聚合为patient-HPO概率，只使用最终bag标签训练，并加入occurrence患者竞争约束。

- E20冻结BiomedBERT向量，fold0仅0.6447；作为E15补充或过滤也均下降。
- E21端到端微调BiomedBERT顶部4层，fold0提高到0.6739（Micro 0.6242、Macro 0.5632），证明潜变量目标可学习，但仍低于E15 fold0的0.6832。
- 对INTRO/DISCUSS/METHODS/CONCLUSION加入可被正bag监督克服的软背景惩罚，fold0仅提高到0.6745。
- E21高置信结果补充E15最多仅0.683256，实际等价于E15且新增项精度不足。
- 即使oracle给出每位患者真实关联数量，再按E15分数Top-K，四折均下降（平均0.6847 vs E15 0.6905），动态cardinality不是当前瓶颈。

结论：软潜变量比硬银标签明显更稳，但现有80篇bag监督不足以学到可泛化的occurrence主体表示；停止继续扩展CLAG或动态Top-K。

## E15多折调参复查

仅使用64篇development pool的四折OOF，不读取locked validation或A榜反馈。系统搜索逻辑回归C、类别权重、基础阈值、Qwen路由区间与融合方式，并独立搜索Marker/SapBERT阈值。

- 原E15四折均值0.69046。
- 关系稳健配置：C=0.02、正类balanced权重乘0.60、基础阈值0.74、Qwen仅用YES救回0.15-0.85区间候选；四折全部正向。
- Marker 0.45、SapBERT 0.90的实体均值从0.75674提高到0.75757，但fold3轻微下降，单独收益很小。
- 合并配置四折为0.68608/0.67238/0.69493/0.71894，均值0.69308，相比E15平均+0.00262；Micro 0.66114、Macro 0.59606。
- 单/多患者分别设阈值没有进一步提升，统一0.74最佳。
- 64篇合并OOF为0.69343 vs 0.69079；1000次配对文档bootstrap差值均值+0.00268，95% CI [-0.00012, 0.00592]，P(提升)=0.971。

结论：发现小幅且四折一致的E15-tuned候选，但置信区间仍轻微跨零，未达到此前约0.01的关系上线门槛。保留为B榜次级候选，不替换E15主方案，也不据此继续细调。

## E23-E25：训练对齐、HPO重排与Marker集成

- E23将关系训练实体从gold替换为逐文档cross-fit预测实体，并比较gold-only、predicted-only、mixed。三者四折均值分别为0.690464、0.690424、0.690464，基本完全相同；训练—推理实体分布对齐未提供增益，停止。
- E24训练上下文HPO交叉编码重排器：输入mention上下文及HPO名称、同义词、definition和父节点；对SapBERT Top-20困难候选打分。固定融合权重alpha=16，gold-span Top-1由0.86434提高到0.88376（+0.01942），四折全部正向。
- E24仅修改非精确词典匹配的预测实体ID，四折共改变86项。结合稳健E15-tuned关系配置后，四折总分0.68566/0.67843/0.69483/0.72300，均值0.69548，相对原E15 CV 0.69046提升+0.00501；Mention 0.72420、Document 0.79370、Micro 0.66452、Macro 0.59948。
- 64篇合并OOF为0.69581 vs 0.69079；1000次配对文档bootstrap差值均值+0.00506，95% CI [0.00088, 0.00922]，P(提升)=0.989。该提升已不跨零，E24保留为新的强候选，但尚未经过线上验证。
- E25训练Marker三个种子2026/3407/7829并平均概率。最佳Marker 0.55、SapBERT 0.90的实体均值0.75764，仅比单种子0.75674提高约0.00091，且fold1下降；不值得三倍Marker推理成本，停止。

### E24全量A榜候选

使用全部80篇标注训练E24（配置保持2 epochs、seed 2026、alpha 16），在E15的1328个实体上仅修改25个非精确词典匹配的HPO ID。生成两份均通过validator 0 error/0 warning的候选：

- 主候选 `submissions/patientphex_a_e24_rerank.jsonl`：E24实体 + 稳健tuned关系（C 0.02、正类权重×0.60、阈值0.74、Qwen仅YES救回0.15-0.85），585关联，SHA256 `2515409baf28000b7a19380e25c9dbfe123d544fc35825b45bb1e5be85969662`。
- 保守消融 `submissions/patientphex_a_e24_entity_only.jsonl`：E24实体 + 原E15关系策略，521关联，SHA256 `5570066018694dd5e55211a1510e25224e76b21b11e9b374bfa999deab6fbd9e`。

两份文件已复制到Windows Downloads。原E15（520关联）保持不变。

E24主候选于2026-09-18线上A榜评估：总分0.6983、Mention 0.7031、Document 0.7751、Micro 0.6505、Macro 0.6644。相比E15线上0.7010/0.7045/0.7758/0.6545/0.6692，分别下降0.0027/0.0014/0.0007/0.0040/0.0048。四项均下降，说明本地HPO重排和关系调参增益未泛化；E24正式否决，不提交同实体的保守消融，线上最佳继续固定为E15。

## E26：重复验证与B榜冻结流水线

- 建立 `data/processed/e26/repeated_splits.json`：固定64篇development、永久排除16篇locked validation，5次重复×4折共20折，每折16篇；首个repeat保留原E0，其余种子3407/7829/104729/99991重新平衡。
- 验收门槛写入划分文件：平均提升至少0.008、20折至少16折正向、paired bootstrap 95% CI必须排除0。
- 新增 `scripts/run_frozen_pipeline.py`，支持对任意未标注PatientPheX输入一键运行E15、E7或两者，自动生成独立实体、Qwen缓存、提交、校验摘要、输入/模型/输出SHA256和manifest。
- 参数化 `train_e14_final.py`、`predict_e3_final.py`、`generate_final_a_entities.py`、`generate_final_a_submission.py`，默认行为保持兼容。
- 首次空目录干跑发现原E15全量Marker未保存导致重训有GPU随机差异；现已冻结 `data/processed/final/marker_e15_model/` 并改为只加载。第二次空目录干跑中E15哈希 `38bdaff7518f4e0bec3e483e0ec197fecfd445ff5cf091e6f95980deaf39f925`、E7哈希 `0b16cb15f639b7654dcab6e1e180e71d81d3d9f0cf4b97c1f2ceac1d9af3dba4` 均与首次完全一致；E7同时精确复现历史提交。

## E22：异构NER侦察

fold0训练PubMedBERT BIO，最佳span F1 0.6487，低于BiomedBERT约0.675。低阈值候选与现有BiomedBERT候选并集后，候选召回仅从0.80246提高到0.80860（+0.00615），新增145个候选仅7个正确（precision 0.0483），远低于预设+0.02门槛；不扩展其余fold。

## 下一步

1. A榜最佳固定为E15（0.7010），保留E7（0.6959）作为次级回退；当前没有新提交。
2. 冻结marker 0.60与SapBERT 0.90，不根据A榜继续调实体阈值。
3. 停止调整E16-E18、段落传播和规则别名；现有bag标签无法可靠监督更细粒度患者归属。
4. 除非获得官方规则澄清或更可靠的原始监督，否则不再消耗A榜提交次数做小幅关系变体。
5. 官方确认association分号、否定关联冲突、异常ID和评价区域。
