# 底座直接判断查阅动作

沿用局部特征实验的 18 条校准答复、28 条留出答复和三版记忆档案。基线复现整句与分句均值分类头，使用 38 条训练答复。另一组直接从同一个孢子候选的语言输出层取得三个单词元选项的分数，未训练额外权重。选项为继续、取消全部和待确认。每条答复轮换三次选项，使各动作依次对应三个字母，按动作对齐后平均分数。两组用相同的校准规则，分别得到阈值。候选均装载第 19、22 层的循环注意力，分类头读取前 12 层，选项组使用完整骨干。 [校准研究](https://arxiv.org/abs/2102.09690) · [澄清决策研究](https://arxiv.org/abs/2604.04565) · [局部特征实验](spora-lookup-action-windows-20261126.md)

| 留出指标 | 分类头 | 底座限定选项 |
| --- | ---: | ---: |
| 校准阈值 | 0.80 | 0.99 |
| 明确指令判对 | 14/24 | 10/24 |
| 明确指令进入待确认 | 3/24 | 13/24 |
| 相反指令两边均判对 | 3/12 | 0/12 |
| 尚未决定时保留待确认 | 4/4 | 4/4 |
| 应读时实际读对 | 48/72 | 60/72 |
| 应取消时仍读取旧记录 | 24/144 | 6/144 |
| 从非空范围开始，取消后清空并结束等待 | 54/108 | 0/108 |

限定选项组在 28 条留出答复上有 14 条在三种选项排列下给出相同的首选动作。「本次调档无须接着办」在三种排列下都被判成继续，经阈值后造成六次实际误读。它在应继续时读回更多值，但其余取消主要进入待确认，已有范围没有清空。当前结果支持将继续和真正撤销分开验收，尤其要检查取消状态与暂缓读回之间的差异。

每条答复跨三组已知地址和四种起始范围，共 336 次离线回放。候选地址、待确认状态及版次由评测提供。分数是三选项内的归一化结果。保留此次固定提示作为对照。后续优先测经过实际入口的完整路径，并将取消正确率及误读作为独立指标。

运行约 140 秒，受检权重和档案摘要一致，35 项相关测试通过。逐题选项分数和档案结果位于 `runs/spora-lookup-backbone-choices-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_action_windows runs/spora-lookup-transitions-clauses-v1 runs/spora-lookup-backbone-choices-new --recurrent @weights --backbone-choices
```
