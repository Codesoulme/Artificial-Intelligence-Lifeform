# 对照循环注意力前后的查阅意图

上一轮的字段重训减少了错提议，新的口语问法仍经常被意图头拦下。检查发现现用探针只计算模型前 12 层，特征没有经过两处循环注意力。借鉴 CheckList 的成对行为测试和 SAGE 的分项评测，本轮固定循环候选、字段头及其阈值，对比前 12 层和前 23 层的特征。前 23 层覆盖两个循环注意力替换。每对话语保留事项，改变执行者、时机或引用范围，例如把「由你核实」换成「由我核实」。两组都用 144 条训练视图，在相同的 48 条校准视图上选择阈值。 [CheckList](https://arxiv.org/abs/2005.04118) · [SAGE](https://arxiv.org/abs/2609.00434) · [上一轮](spora-lookup-schema-ranking-20261126.md)

新留出使用新姓名、新措辞和四种字段。每行分别保留两个分句的意图标签，方便区分正确识别双请求和误判成两个都不用查。当前评测只允许提议一条记录。

| 指标 | 旧意图头 | 新头，前 12 层 | 新头，前 23 层 |
| --- | ---: | ---: | ---: |
| 校准集单项正确提议 | 12/24 | 24/24 | 16/24 |
| 新留出单项正确提议 | 14/32 | 8/32 | 8/32 |
| 新留出单项错提议 | 2 | 2 | 2 |
| 新留出零项的多余提议 | 2/16 | 0/16 | 0/16 |
| 新留出双项被误判为单项 | 2/16 | 4/16 | 4/16 |
| 新留出两个分句均判对 | 40/64 | 38/64 | 38/64 |
| 原第二套措辞单项正确提议 | 8/16 | 8/16 | 8/16 |
| 原暂缓与改写的多余提议 | 2/8 | 0/8 | 0/8 |

两个新头均从校准集选到 0.99。它们在原第一批留出的正确提议也都从 24/24 降到 20/24。「把之前留底的……讲给我听」这一组在三种条件下都没有正确提议。仅按单项选中结果计算会把部分双项漏判计作正确拒绝，表中另列两个分句都判对的数值。本轮同时改变特征深度和后续层类型，仅支持比较两个读出位置，尚不能单独估计循环替换的影响。下轮可在相同深度对照父检查点与循环候选，并以单项漏报、双项漏报及零项误报共同校准。

首次深层运行触发了循环层只支持单条无填充输入的限制。改用逐条提取并在输出端补齐后，完整运行约 102 秒，35 项相关测试通过，受检输入权重摘要一致。新权重和逐题结果位于 `runs/spora-lookup-intent-depth-v2`。两个新头均处于离线实验阶段。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_intent_depth runs/spora-switch-address-matched-v1 runs/spora-lookup-field-focus-independent-v1 runs/spora-lookup-schema-ranking-v1 runs/spora-lookup-intent-depth-new --recurrent @weights
```
