# 第 24 层保留两个精确注意力头的对照

定向蒸馏让三层循环版恢复了一条中等长度更正题，最长更正与公开式候选问答仍落后。Retrieval-Aware Distillation 在其他模型上发现少数注意力头对上下文检索很重要。本轮冻结刚完成定向蒸馏的三层权重，只把第 24 层的部分循环输出换成对相同历史的精确注意力输出。 [定向蒸馏](spora-hybrid-targeted-recovery-20261126.md) · [Retrieval-Aware Distillation](https://arxiv.org/abs/2602.11374)

从训练分区选最长的四对更正请求，每对只改一个历史值。对每个头单独启用精确注意力，以正确值首次分歧词元相对另一值的平均分差增益排序。全部 12 个头恢复后，与相同第 19、22 层权重、原生第 24 层的对照在八条请求上的分差最大差异为 0.0156。第 9、8 号头提升最多，平均增益分别为 0.105、0.045。第 11、7 号头作为同开销对照，两组各用两个查询头，读取相同的一个键值组。头号从零开始。

| 第 24 层配置 | 约 700 词元更正 | 约 1,400 词元更正 | 两种长度合计 | 开放题逐轮 |
| --- | ---: | ---: | ---: | ---: |
| 三层定向训练，全部循环 | 6/10 | 1/10 | 7/20 | 3/18 |
| 第 9、8 号头精确读取 | 6/10 | 1/10 | 7/20 | 4/18 |
| 第 11、7 号头精确读取 | 5/10 | 2/10 | 7/20 | 4/18 |

两组都补回了同一个编辑对话的末轮。分数较高的两个头没有在独立更正题中表现出优势。分差考察的是正确值的首次分歧，实际生成还需保留第二个事实并正常结束。实验用时约 507 秒，源文件与个体状态摘要一致。实验钩子保留原循环状态，同时独立保存选中键值组，未测部署后的存储与速度。CPU 对照验证选中头的整段及增量输出与精确注意力一致，其他头与循环对照一致。

下一轮优先检验生成分布。GenDistill 显示，概率评测会低估混合模型实际生成的差距。OPAL 在学生自己生成的前缀上让冻结教师给出逐词元分布，用反向 KL 更新。孢子早期在查询键投影上做过类似试验，这次比较固定三层循环布局下的相同更新预算，控制前缀来源，仍以独立生成和持久读取验收。 [GenDistill](https://arxiv.org/abs/2603.26556) · [OPAL](https://arxiv.org/abs/2609.31947) · [早期查询投影实验](spora-query-key-student-prefix-20261122.md)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$run = 'runs/spora-hybrid-targeted-recovery-v1'
$layers = @("$run/targeted/layer-19", "$run/targeted/layer-22", "$run/targeted/layer-24")
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_recurrent_retrieval_heads runs/spora-composition-late-gates-20-23-24-v1/candidate.pt "$run/result.json" runs/spora-recurrent-retrieval-heads-v1 @layers
```
