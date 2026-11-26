# 分句独立判断查阅意图

上一轮的分句选择能定位所问事项，整句意图模型却挡下全部请求。本轮在同一批 48 条训练视图上，将分句选择改为逐项二分类。每个分句独立判断是否要求查阅。只有恰好一项超过阈值时，才用原字段头排序并核对目录。两项都暂缓或都要查时，实验不提议单个地址。骨干、字段权重和字段阈值沿用上一轮。 [上一轮实验](spora-lookup-field-focus-20261126.md) · [PassiveQA](https://arxiv.org/abs/2604.04565) · [SAGE](https://arxiv.org/abs/2609.00434)

24 道校准题含 16 条单项请求和 8 条零项或双项请求，全部判断正确，校准阈值为 0.5。校准使用独立人名和措辞。评测使用原留出、暂缓及改写对照，以及另一批由不同措辞生成的单项、双项和零项请求。

| 评测 | 正确判断单项或不提单项 | 正确地址提议 | 错误或多余提议 |
| --- | ---: | ---: | ---: |
| 原留出单项请求 | 24/24 | 12/24 | 8 |
| 原暂缓与改写对照 | 6/8 | — | 2 |
| 新措辞单项请求 | 8/16 | 4/16 | 4 |
| 新措辞零项或双项请求 | 8/8 | — | 0 |

在原留出中，上一轮任一条件的正确地址提议均为零。新头把十二条用品请求带到正确提议，书目中四条选对字段但未通过字段阈值，八条错判为用品或饮品且通过阈值。新留出中的四次错提议也来自书目。逐条移除正确地址后，这十二次错误仍可通过，因为错误首选属于目录中的其他字段。两条多余提议来自「我会另外去确认」这一暂缓措辞，按两个人名重复测试。

结果支持继续单测书目字段，同时补足暂缓意图的语义对照。评测仅使用候选目录和地址提议。训练产物保留在实验目录，未进入聊天入口。运行约 91 秒，输入权重摘要前后一致。逐题结果及权重位于 `runs/spora-lookup-field-focus-independent-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_field_focus runs/spora-switch-address-matched-v1 runs/spora-lookup-field-focus-independent-new --recurrent @weights
```
