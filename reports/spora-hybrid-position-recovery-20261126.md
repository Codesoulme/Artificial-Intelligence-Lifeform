# 第三处循环迁移的新增位置对照

第 24 层的精确历史实验表明，完整保留事实并提高读取权重后，最长更正题的通过数仍为 2/10。本轮保留第 19、22 层作为共同布局，分别追加第 13、20、25 层。三个位置覆盖较早的骨干、两处循环之间和靠近输出的位置。FlashMorph2 建议在实际离散组合和恢复训练后评价布局。GenDistill 发现混合模型的答复概率和实际生成的排名会不同。 [保留与读取实验](spora-selective-recurrent-history-20261126.md) · [FlashMorph2](https://arxiv.org/abs/2611.24809) · [GenDistill](https://arxiv.org/abs/2603.26556)

## 单层对齐

三处新增位置与既有第 21、24 层使用同样的 36 篇校准短文、24 条留出、8 词元块、160 步局部输出对齐和 32 步预测对齐。循环更新沿用求和规则。单独换一层时，第 25 层最接近完整注意力父模型的预测分布。

| 被替换层 | 对齐后 KL | 留出首选词元与教师一致 | 相对混合误差 |
| --- | ---: | ---: | ---: |
| 13 | 0.10231 | 23/24 | 0.06619 |
| 20 | 0.30354 | 20/24 | 0.15504 |
| 21，既有对照 | 0.31215 | 21/24 | 0.18360 |
| 24，既有对照 | 0.03735 | 24/24 | 0.26119 |
| 25 | 0.02548 | 24/24 | 0.22120 |

## 实际布局恢复

双层和三个新增布局分别在相同的 24 段 CrossWOZ 训练对话上更新 24 次，共 665 个监督词元。只更新循环核，损失为标准答复交叉熵与完整注意力父模型输出 KL 的均值。48 段官方测试对话共 1,115 个答复词元。四组的训练顺序、学习率和初始共同层权重相同。重新运行的双层基线在答复概率评测中与上次逐题一致。合计训练及概率评测约 196 秒，CUDA 张量训练峰值约 3.66 GiB。 [既有恢复对照](spora-hybrid-language-recovery-20261126.md)

| 循环层 | 恢复前 NLL | 恢复后 NLL | 恢复后首选词元准确率 |
| --- | ---: | ---: | ---: |
| 完整注意力父模型 | 4.88416 | 4.88416 | 30.22% |
| 19、22 | 4.96425 | 4.94577 | 28.61% |
| 13、19、22 | 4.97734 | 4.95239 | 28.16% |
| 19、20、22 | 5.01766 | 4.99483 | 27.71% |
| 19、22、25 | 5.03578 | 4.99847 | 27.80% |

第 25 层在单层对齐中领先，装入现有布局后，恢复前和恢复后的答复 NLL 都排在本轮三个新候选的最后。第 13 层的恢复后 NLL 距双层差 0.00662，首选词元准确率差约 0.45 个百分点。

## 重载后的生成

以独立进程加载训练保存的每层权重，并核对恢复报告里的摘要。评估包含原有 30 条更正问题、18 轮开放对话、32 条受控具名读回，以及另外五组未用于训练的中文对话。开放对话把每次实际答复放入下一轮历史。每轮使用 48 词元的生成上限，按明确的必备信息、应排除信息及正常结束判分。四组重载及生成评估约 408 秒。

| 循环层 | 短更正 | 约 700 词元 | 约 1,400 词元 | 原开放对话逐轮 | 新开放对话逐轮 | 具名读回 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 19、22 | 8/10 | 7/10 | 2/10 | 8/18 | 2/15 | 32/32 |
| 13、19、22 | 10/10 | 8/10 | 3/10 | 7/18 | 3/15 | 32/32 |
| 19、20、22 | 6/10 | 2/10 | 1/10 | 5/18 | 3/15 | 32/32 |
| 19、22、25 | 8/10 | 5/10 | 4/10 | 8/18 | 2/15 | 32/32 |

第 13 层组合在三个长度上比双层基线多答对 4 条，原开放对话少过一轮。第 25 层组合在最长前文里正确输出更新值的题数从 3/10 升到 5/10，完整答对数从 2/10 升到 4/10，约 700 词元组却由 7/10 降到 5/10。四种布局的具名读回逐题及结束情况一致。现阶段第 13、25 层分别提供较均衡的更正表现和最长前文优势，值得在另一套长对话中复核。

完整注意力父模型在新开放对话中通过 3/15，四个迁移版本在新题的 15 次生成均正常结束。人工浏览可见，模型把提前一小时误改成提前半小时，漏掉通知中的日期，也会选错符合用户要求的照片。词面判分也有局限，例如「选装松木椅子」选对了椅子，答复缺少评分要求的保修理由。新题是小样本行为检查。

## 原始底座和生成上限

同一台机器、同一分词器及同样的直接答复模板下，原始 DeepSeek-R1-Distill-Qwen-1.5B 使用 48 词元上限时通过 2/15，9 次正常结束。将上限设为 96 后通过 3/15，12 次正常结束。完整注意力孢子父模型在两个上限下都通过 3/15，15 次正常结束，最长答复是 21 词元。补测约 136 秒，逐题发言存于 `runs/spora-hybrid-position-recovery-v1/transfer-controls.json`。在当前解码方式下，开放对话失误在原始底座上也存在，父模型的得分没有随生成上限增加而提高。官方建议保留推理并使用采样，这里的直接贪心生成用于控制布局比较。 [DeepSeek 官方使用说明](https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations)

接下来用新场景复核第 13、25 层与双层布局。普通对话还有一个待分离的问题。MT-Eval 将多轮任务与约束相等的单轮版本配对，并发现前轮错误会影响后续。可对新开放题用模型自己的历史、正确历史和汇总为单句的请求作对照，进一步定位底座能力与承接错误的影响。 [MT-Eval](https://arxiv.org/abs/2401.16745) · [多轮对话研究综述](https://arxiv.org/abs/2504.04717)

为后续扩大中文评估，在本机获取并核验了 Multi-IF 的公开快照 `0ab97ce`。其中有 454 组中文三轮请求，带逐轮约束及验证参数。后续可从支持本地确定性校验的约束中选固定样本，验证中文开放题的结果是否具有代表性。当前保存在本地 `runs/research/multi-if-0ab97ce`，数据使用 CC BY-NC 2.0。 [Multi-IF 论文](https://arxiv.org/abs/2410.15553) · [数据快照](https://huggingface.co/datasets/facebook/Multi-IF/tree/0ab97ce0b45c7f57772e8ba2ac1616f4b00bd3aa)

## 新场景复核

模型和解码参数固定，换用五个新场景及两种取值，复用相同长度的干扰对话。中等组为 702–717 词元，最长组为 1,404–1,419 词元。三组重新加载与评估约 259 秒，长更正的 60 条答复均正常结束，具名读回均为 32/32。

| 循环层 | 约 700 词元完整 / 含新值 / 含旧值 | 约 1,400 词元完整 / 含新值 / 含旧值 |
| --- | ---: | ---: |
| 19、22 | 0/10 · 0/10 · 6/10 | 0/10 · 1/10 · 2/10 |
| 13、19、22 | 1/10 · 1/10 · 3/10 | 0/10 · 0/10 · 4/10 |
| 19、22、25 | 0/10 · 0/10 · 6/10 | 0/10 · 1/10 · 1/10 |

第 13、25 层的优势在新场景中明显缩小。自动通过的一条答对了录音时间和剪辑员，但开头额外写了相冲突的「周六」。常见错误是复述旧值，例如读到了导览牌的负责人，却仍写旧摆放点「松街校门」。目前的主要问题是如何把新事实选出来并据此生成，在这批题上单靠调整替换位置无法解决。

原始 DeepSeek 在相同新题上，短、约 700、约 1,400 词元三个长度的完整更正均为 0/10，正常结束分别为 6、5、6 次。完整注意力孢子父模型分别为 4/10、1/10、0/10，30 条生成均正常结束。父模型答出新值的次数随长度从 9/10 降到 2/10、1/10。这条长度退化在替换循环注意力前已经存在。对照使用相同直接贪心解码和 48 词元预算，耗时约 205 秒，逐题输出见本地 `recall-transfer-controls.json`。当前继续以第 19、22 层作为基线。下一步将长更正拆成近距离与远距离、单字段与双字段的对照，在完整注意力父模型中先查清旧值复述和新值遗漏，再用同样条件衡量循环迁移的附加损失。 [LongBench Pro](https://arxiv.org/abs/2601.02872) · [RECON](https://arxiv.org/abs/2607.16716)

37 项循环注意力、对话注意力一致性与本地记忆读回测试通过。中文公开评测的初筛发现，Multi-IF 中有 20 组三轮对话只包含已核对验证方式的标题、关键词、结尾、引号和逗号约束，适合作为补充检查。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$checkpoint = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$revision = 'runs/spora-gelu24-write-revision-v1/comparison.json'
foreach ($layer in (13,20,25)) {
    .\.venv\Scripts\python.exe -u -m scripts.audit_spora_recurrent_attention $checkpoint $revision "runs/spora-gelu24-recurrent-layer${layer}-morph-v1" --layer $layer
    if ($LASTEXITCODE -ne 0) { break }
}
$layers = @('runs/spora-gelu24-recurrent-layer13-morph-v1', 'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer20-morph-v1', 'runs/spora-gelu24-recurrent-layer22-v1', 'runs/spora-gelu24-recurrent-layer25-morph-v1')
$out = 'runs/spora-hybrid-position-recovery-v1'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_hybrid_language_recovery $checkpoint runs/research/crosswoz-df82c9f/train.json.zip runs/research/crosswoz-df82c9f/test.json.zip $out @layers --layout '19,22' --layout '13,19,22' --layout '19,20,22' --layout '19,22,25'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_recovery $checkpoint $out "$out/generation.json" --transfer data/recurrent-layout-transfer-dialogue-20261126.json
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_recovery $checkpoint $out "$out/recall-transfer.json" --recall-transfer data/recurrent-layout-recall-transfer-20261126.json --layout '19,22' --layout '13,19,22' --layout '19,22,25'
```
