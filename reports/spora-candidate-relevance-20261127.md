# 逐字段匹配与无记录弃权

上一轮直接选两个地址时，两道无匹配题都读了其他场景的记录。本轮把四个候选分别交给二分类头，允许结果包含零项或一项。Schema-Guided Dialogue 用字段描述支持跨场景迁移。StaleBench 和 SAGE 分别说明了独立检查寻址及弃权的价值。 [组合寻址对照](spora-grouped-revision-routes-20261127.md) · [Schema-Guided Dialogue](https://arxiv.org/abs/1909.05855) · [StaleBench](https://arxiv.org/abs/2611.19752) · [SAGE](https://arxiv.org/abs/2609.00434)

冻结双循环层骨干，提取第 23 层之后的末位特征，使用同一个小型分类头给每个请求和候选字段对打分。输入只含末问和字段描述。训练 48 条请求，校准 24 条，主要留出 32 条，另复核上一轮的 8 条原题。三套合成数据分别使用不同场景、问法与缺失字段描述。候选中含同一场景的其他字段，也含其他场景的字段。对照按请求与字段名称共有的相邻字符对打分。两种方法各自用校准集的整题准确数减去多选的错误字段数选择阈值，平分时优先误选少的阈值。

| 已知应读数量时，正确字段得分均高于干扰项 | 神经匹配 | 字符重合 |
| --- | ---: | ---: |
| 留出双字段 | 10/16 | 8/16 |
| 留出单字段 | 5/8 | 1/8 |
| 上轮有匹配的原题 | 4/6 | 3/6 |

两种方法最终都选到 1.01，即拒绝所有候选。上表只衡量排序，不能证明模型自行识别出应读数量。为解释排序与实际弃权的差距，下表用已保存的神经分数检查固定阈值，未在留出集上另选部署阈值。

| 阈值 | 留出正确选中，40 个应读字段 | 留出误选字段 | 其中无匹配题误选 |
| --- | ---: | ---: | ---: |
| 0.50 | 35 | 31 | 7 |
| 0.90 | 22 | 11 | 1 |
| 0.95 | 16 | 4 | 0 |
| 0.99 | 7 | 2 | 0 |
| 校准使用的 1.01 | 0 | 0 | 0 |

初跑的三个划分复用了缺失字段短语。在拆开这些措辞后重新训练，初跑仅 4/40 的留出读取进一步降为 0/40。最终实验约 135 秒，源权重摘要一致，32 项相关回归通过。该轮只验证选址，未读取持久值或生成答复。候选保存在 `runs/spora-candidate-relevance-v2`，未接入入口。下一轮优先给逐字段匹配补上同一请求下的正负排序监督，并单独用近义字段和错场景反例验证。有效记录已确认后的成句训练可独立推进。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$parent='runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$layout='runs/spora-hybrid-position-recovery-v1/19-22'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_candidate_relevance $parent runs/spora-candidate-relevance-new "$layout/layer-19" "$layout/layer-22"
```
