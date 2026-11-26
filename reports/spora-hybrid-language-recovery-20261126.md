# 三种循环布局的中文预测恢复

上一轮以 48 段真实中文服务对话衡量三层循环迁移的语言损失。FlashMorph2 的整网离散布局评估和恢复目标联合适配说明，初始误差不足以决定最后采用的层位置。HALO 与 MOHAWK 也报告了局部对齐后再做整网预测蒸馏的价值。前者在 Qwen3 系列上的转换耗费约 23 亿词元，本地试验是小预算方向验证。 [FlashMorph2](https://arxiv.org/abs/2611.24809) · [HALO](https://arxiv.org/abs/2601.22156) · [MOHAWK](https://arxiv.org/abs/2408.10189)

## 实验

使用固定版本 CrossWOZ 官方训练分区按会话编号摘要取 24 段，各包含三轮对话，第三轮系统答复共 665 个监督词元。三组的顺序、样本和 24 次 AdamW 更新完全一致，学习率 0.0005。损失为真实答复交叉熵和完整注意力父检查点的预测 KL 等权平均。整网执行实际离散结构，只有循环核参与梯度更新。原个体摘要及源检查点和四份迁移候选的文件摘要保持一致。

测试沿用上一轮 48 段官方测试对话及 1,115 个答复词元，训练与测试会话编号无交集。CrossWOZ 属于任务型对话，结果仅能用于衡量此分布下的语言保真程度。

## 结果

| 循环布局 | 恢复前 NLL | 恢复后 NLL | 恢复前首选词元 | 恢复后首选词元 |
| --- | ---: | ---: | ---: | ---: |
| 完整注意力父模型 | 4.8842 | 4.8842 | 30.22% | 30.22% |
| 19、22 | 4.9642 | 4.9458 | 28.34% | 28.61% |
| 19、21、22 | 4.9736 | 4.9464 | 27.80% | 27.80% |
| 19、22、24 | 4.9760 | 4.9518 | 27.53% | 27.89% |

三种布局合计训练及预测评估约 150 秒，PyTorch 训练峰值分配约 3.65 GiB。两种第三层候选均降低了语言损失，其中第 21 层候选恢复后与双层版平均 NLL 相差 0.0006，首选词元准确率仍差约 0.81 个百分点。这一小样本上的损失差异不足以判断总体语言能力。10 项循环注意力单元测试通过。

## 重新加载后的实际生成

在独立进程加载三组训练产物，沿用长前文更正题及开放中文多轮词面题，额外使用相同的具名当前值及历史值读取评估。

| 循环布局 | 短题完整 | 约 700 词元完整 | 约 1,400 词元完整 | 开放题逐轮通过 | 具名读取精确通过 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 19、22 | 8/10 | 7/10 | 2/10 | 8/18 | 32/32 |
| 19、21、22 | 4/10 | 0/10 | 1/10 | 6/18 | 32/32 |
| 19、22、24 | 8/10 | 5/10 | 1/10 | 3/18 | 32/32 |

评估用时约 276 秒，全部读取精确通过且正常结束，个体摘要与恢复训练完全一致。读取题使用受控写入形成当前和历史值，尚未覆盖任意开放事实。双层恢复后的更正和开放题通过数与上一轮原始双层基线相同；两种第三层候选均出现更明显的实际多轮退化。暂时保留双层布局。

下一轮需要诊断候选的长前文信息传递和恢复训练监督范围。Balancing Memory Pathways 指出，普通监督可能增加对精确注意力的依赖。孢子此前直接使用分段辅助损失后，受限条件下的远距读取仍为 0/10，重复相同的短期训练难以解决路径瓶颈。结合本轮短答案预测和长答复的明显分歧，下一步测量第三层迁移时的逐层信息失真，并比较对齐完整长前文的教师隐藏状态与仅对齐答案概率的收益。 [循环路径已有结果](spora-recurrent-pathways-20261125.md) · [Balancing Memory Pathways](https://arxiv.org/abs/2610.06750) · [MOHAWK](https://arxiv.org/abs/2408.10189)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$parent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$recovery = 'runs/spora-hybrid-language-recovery-v1'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_hybrid_language_recovery $parent runs/research/crosswoz-df82c9f/train.json.zip runs/research/crosswoz-df82c9f/test.json.zip $recovery runs/spora-gelu24-recurrent-layer19-v2 runs/spora-gelu24-recurrent-layer21-morph-v1 runs/spora-gelu24-recurrent-layer22-v1 runs/spora-gelu24-recurrent-layer24-morph-v1 --steps 24 --count 48
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_recovery $parent $recovery "$recovery/generation.json"
```

逐步损失、独立测试逐题结果及可重新加载的循环核位于 `runs/spora-hybrid-language-recovery-v1`。
