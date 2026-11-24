# 冻结循环历史后调整末层对话读出

从两层循环注意力的同一检查点起步。用六种新场景、每种两套值和约 1,400 词元前文，先做普通对话残差开关对照。仅切换普通对话路径，当前个体和两个循环注意力权重均冻结。对照经校验使用相同的输入、生成预算和读取状态。

| 模型 | 普通对话残差 | 完整回答 | 新值出现 | 保留字段出现 | 正常结束 |
| --- | --- | ---: | ---: | ---: | ---: |
| 完整注意力父模型 | 启用 | 1/12 | 4/12 | 6/12 | 12/12 |
| 完整注意力父模型 | 关闭 | 0/12 | 0/12 | 1/12 | 7/12 |
| 两层循环注意力 | 启用 | 1/12 | 4/12 | 5/12 | 12/12 |
| 两层循环注意力 | 关闭 | 0/12 | 1/12 | 5/12 | 4/12 |

关闭时完整通过与正常结束均下降。现有残差至少有助于这批输入的语言输出。该开关一次影响多层输出，不能定位某一层的作用。逐题结果位于 `runs/spora-recurrent-split_gated_block_delta-warm-v1/dialogue-residual-ablation.json`。

## 末层训练

保留第 27 层现有的秩 16 普通对话残差作为起点，仅更新该层的输入与输出投影。循环注意力、语言底座、词表投影和个体冻结。利用底座已预填充的完整前文，在最后一层残差之前保存答案位置的隐藏向量。在缓存的短向量上重复训练，首条记录检查该路径与完整前向的词表分数一致到 FP16 容差。该处理借鉴在冻结底座上调整隐藏表示的方向。 [ReFT](https://arxiv.org/abs/2404.03592)

72 条目标回答来自 36 个成对更正场景，只用已有训练分区构造。另有 20 条基础中文问答用于保持已有生成。192 次更新中，四分之三使用完整答案和字段均衡损失，四分之一匹配起点在普通问答首个词元和结束位置的概率分布。训练答案包含结束符。生成验证由本地孢子解码器逐词完成，测试答案不进入前文。

| 指标 | 起点 | 末层续训 |
| --- | ---: | ---: |
| 原更正题，约 700 词元 | 5/10 | 7/10 |
| 原更正题，约 1,400 词元 | 4/10 | 3/10 |
| 六类新场景，约 1,400 词元 | 1/12 | 1/12 |
| 开放题规则通过，整组 / 逐轮 | 2/6 · 9/18 | 1/6 · 5/18 |
| 受控持久读取 | 32/32 | 32/32 |

新场景的两份结果逐题校验为同一历史和问题。旧题约 700 词元时，续训比起点少提两次旧值，多答对两题。最长组和开放续聊出现退步。训练向量基于参考答案前缀，开放测试使用模型实际生成的前缀。两次受控持久读取逐题输出一致，个体摘要一致。末层参数在试验目录独立保存，继续使用原检查点作为基线。采集、训练和评测共约 349 秒，训练阶段 CUDA 峰值分配为 3.40 GiB。 [前一轮的字段对照](spora-recurrent-field-composition-20261124.md)

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$sporaCandidate = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
$sporaParent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_value_preferences $sporaParent "$sporaCandidate/dialogue-residual-ablation.json" "$sporaCandidate/layer-19" "$sporaCandidate/layer-22" --dialogue-ablation data/recurrent-phase-transfer-v1.json
.\.venv\Scripts\python.exe -m scripts.train_spora_recurrent_readout $sporaParent runs/spora-recurrent-final-readout-v1 "$sporaCandidate/layer-19" "$sporaCandidate/layer-22"
```

## 在同一问题中列出字段

回到续训前的循环检查点。保留原问题和对话历史，比较直接回答、附加两行格式要求、再把两个字段明确列成小问。提示中只列请求，没有给出正确值。本轮的通过规则仍要求包含两个正确值、排除旧值并正常结束。两行形式本身不计入分数。 [分解任务的提示研究](https://arxiv.org/abs/2205.10625)

| 提问方式 | 完整回答 | 新值出现 | 旧值出现 | 保留字段出现 |
| --- | ---: | ---: | ---: | ---: |
| 原问题 | 1/12 | 4/12 | 2/12 | 5/12 |
| 加两行格式要求 | 1/12 | 2/12 | 5/12 | 7/12 |
| 加格式要求并列出两项 | 0/12 | 4/12 | 5/12 | 4/12 |

36 条答复均正常结束。这组提示延长了最终问句，并未把两个读取请求分开执行。原对照中，独立调用可分别取回两项的题有 7/12。接下来应在生成前测量两次内部读取能否分别取到值，以及把各自的隐藏表示交给同一个解码器后能否保留两项。将实验失败的回答留在对照目录，正式检查点继续使用原版。逐题输出位于 `runs/spora-recurrent-split_gated_block_delta-warm-v1/answer-plan.json`。

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_field_composition $sporaParent "$sporaCandidate/answer-plan.json" "$sporaCandidate/layer-19" "$sporaCandidate/layer-22" --parent-report "$sporaCandidate/far-transfer-parent.json" --recurrent-report "$sporaCandidate/far-transfer.json" --answer-plan
```
