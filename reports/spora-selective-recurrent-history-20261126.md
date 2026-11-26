# 第 24 层的固定容量精确历史

此前只恢复两个精确注意力头，未改善更正题总分。本轮把容量限制放在历史条目数上。第 24 层保留原来的循环矩阵与近期窗口，当一块 8 词元移出近期窗口时，用已有的带门控增量规则更新矩阵，并将该块的原始键值加入候选。保留策略按各词元的预测残差范数乘写入门值打分。在共享键值的查询头间取平均，每个原生键值头只保留 32 条。保留条目与近期条目一起参加因果注意力，循环输出沿用原模型的门控混合。这是对 HOLA 写入思路的块级实现，读取侧沿用孢子现有点积和混合门。 [HOLA](https://arxiv.org/abs/2607.02303) · [注意力头对照](spora-recurrent-retrieval-heads-20261126.md)

三组共享同一父检查点和第 19、22 层初始循环权重，第 24 层使用同一组已对齐的门控增量权重。按近期保留组与残差组的条目和字节预算相同。最长输入处两组额外状态各占 33,536 字节，包括 FP16 键值、位置和分数。保留组用有界分块前向，纯循环组沿用原批处理前向。三组合计用时约 321 秒，输入来源与个体摘要均已核对。

| 第 24 层方案 | 短更正 | 约 700 词元更正 | 约 1,400 词元更正 | 开放多轮逐轮 | 完整开放对话 | 具名读取 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 纯循环 | 8/10 | 5/10 | 2/10 | 4/18 | 0/6 | 32/32 |
| 按最近时间保留 | 7/10 | 5/10 | 1/10 | 7/18 | 2/6 | 32/32 |
| 按写入残差保留 | 8/10 | 5/10 | 2/10 | 7/18 | 2/6 | 32/32 |

两种保留策略通过了完全相同的开放轮次，具名读回逐条一致。按残差保留在最长前文中仍能保留位置很早的条目，也在 2/10 条题中留下新值的一部分词元。约 700 词元组，新值和另一字段至少留下一词元的题数分别为 4/10 和 2/10。两种长度下，残差组在任一键值头中完整留住任一所需字段的题数均为 0/10。近期组在两种长度下都未覆盖所需字段。字段位置由聊天提示词的分词偏移离线标注，只用于统计。

## 验证保留与读取的瓶颈

在相同预算上，优先保留新值和另一字段的完整词元，剩余槽位仍按残差挑选。对照优先保留等量的早期无关词元。逐题核对两组优先位置确实保留到生成开始，并经过相同读取路径。强制保留是一项使用题目字段标注的诊断。它测量精确历史中已有证据时的生成行为。 [KVDiagnosis](https://arxiv.org/abs/2608.09412) · [语义单元保留](https://arxiv.org/abs/2502.01941)

| 方案 | 约 700 词元完整更正 | 约 1,400 词元完整更正 | 最长题新值出现 | 最长题另一字段出现 |
| --- | ---: | ---: | ---: | ---: |
| 残差保留 | 5/10 | 2/10 | 2/10 | 4/10 |
| 优先保留无关词元 | 6/10 | 1/10 | 2/10 | 3/10 |
| 优先保留完整事实 | 7/10 | 2/10 | 2/10 | 5/10 |

完整事实在每道题的两个键值组中都保留到了生成开始。它使约 700 词元的更正比自动残差选择多通过两题。约 1,400 词元时，新值仍只被答出两次，说明精确条目中的事实被保留后，现有读取路径仍难以转化成完整答复。两组共约 156 秒。

保持同一批槽位和优先位置，把优先条目在精确注意力中的 softmax 权重乘以 4，再做归一化。最长题中，优先完整事实组仍为 2/10，优先无关词元组仍为 1/10。完整事实组多答出一次新值，但整题通过数没变。该复测用时约 102 秒，逐题槽位与干预前一致。局部读取加权的结果支持换一个第三层位置再测。下一轮对中部、现有两处循环之间以及靠近输出的候选做同预算对齐，再在真实的三层布局上比较生成与独立留出。 [FlashMorph2](https://arxiv.org/abs/2611.24809) · [生成评测与混合结构蒸馏](https://arxiv.org/abs/2603.26556)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$layers = @('runs/spora-gelu24-recurrent-layer19-v2','runs/spora-gelu24-recurrent-layer22-v1','runs/spora-gelu24-recurrent-layer24-gated-morph-v1')
$checkpoint = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_selective_recurrent_history $checkpoint runs/spora-selective-recurrent-history-v1 @layers --slots 32
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_selective_recurrent_history $checkpoint runs/spora-selective-history-evidence-v1 @layers --slots 32 --evidence-reference runs/spora-selective-recurrent-history-v1/result.json
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_selective_recurrent_history $checkpoint runs/spora-selective-history-read-boost-v1 @layers --slots 32 --evidence-reference runs/spora-selective-recurrent-history-v1/result.json --read-multiplier 4
```

37 项循环注意力、对话注意力一致性与持久化测试通过。测试覆盖固定容量、分段一致性、保留区与近期区分离，以及历史全部能放入保留区时的因果注意力一致性。逐题生成与实际槽位在本地 `runs/spora-selective-recurrent-history-v1`。
