# 保存待确认动作后的查阅回放

上一轮让模型从提问和答复全文预测最终动作，偶尔把取消判成继续。本轮把提问时已知的动作保存在待确认状态里。模型只看用户答复，识别同意、反对、直接继续、直接取消或尚未决定。状态更新将同意作用于已保存的动作，将反对作用于它的相反动作。直接指定的动作覆盖提问中的选项。评测借鉴 ToolTurns 和 StateMemBench，将状态变化和档案读回分开计分。 [ToolTurns](https://arxiv.org/abs/2611.14925) · [StateMemBench](https://arxiv.org/abs/2608.19652) · [上轮实验](spora-lookup-confirmation-20261126.md)

读出训练使用 13 种独立答复，校准使用另外 8 种。两组均选到 0.34，分别读取前 12 层和前 23 层特征。前 23 层经过两处循环注意力替换。每套评测含 40 个问答组合、三组档案地址和四种起始查阅范围，共 480 次回放。两套分别为上轮留出和本轮新措辞。回放从已建立的待确认状态开始，由实验提供候选地址、提问所指动作及版次，随后核对实际读出的记录和值。

| 指标 | 旧题，前 12 层 | 旧题，前 23 层 | 新题，前 12 层 | 新题，前 23 层 |
| --- | ---: | ---: | ---: | ---: |
| 肯定与否定判对 | 12/16 | 12/16 | 16/16 | 16/16 |
| 明确指定动作判对 | 16/16 | 10/16 | 16/16 | 12/16 |
| 尚未决定却被清空 | 6/8 | 2/8 | 4/8 | 4/8 |
| 从非空范围开始，取消后清空且结束等待 | 126/144 | 108/144 | 144/144 | 108/144 |
| 取消后仍读取 | 12/192 | 12/192 | 0/192 | 24/192 |
| 应读时读对记录、版次和值 | 84/96 | 60/96 | 96/96 | 96/96 |

上轮前 12 层整句读出在旧题正确读回 42/96，取消后仍读取 24 次。新旧方案的输入和监督标签均有差异，这个对照反映整套方案的变化。新题前 12 层的读回指标全部正确，但它把「同伴同意了，我还在考虑」判成取消，造成四个问答组合提前清空。只统计是否读出旧值会漏掉这种错误。当前权重留在离线实验，继续用独立转述和犹豫措辞验证待确认状态能否正确保留。

完整运行约 122 秒，受检权重与档案摘要一致。35 项相关测试通过，另核对了各划分的答复互不重叠，以及同一句肯定或否定在相反提问下对应相反动作。逐条结果和两份读出权重位于 `runs/spora-lookup-stored-confirmation-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_confirmation runs/spora-lookup-transitions-clauses-v1 runs/spora-lookup-reset-matched-v1 runs/spora-lookup-stored-confirmation-new --recurrent @weights --stored-action
```
