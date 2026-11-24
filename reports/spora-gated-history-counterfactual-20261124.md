# 门控循环历史的成对更正训练

从已通过重载复核的第 19、22 层门控循环注意力起步。每个训练对保留相同的登记、问题和干扰段落，改变更正后的值。训练覆盖完整答案和结束标记，并在两条正确答案首次分歧的位置压低另一条答案的词元。底座和个体冻结。这轮实际抽到 31 个不同的训练对，共更新 32 步。训练及评测约 642 秒，训练时 PyTorch 峰值分配为 4.15 GiB。 [上一轮门控结果](spora-recurrent-gated-delta-20261124.md)

| 留出指标 | 起点 | 成对训练后 |
| --- | ---: | ---: |
| 约 700 词元成对题，单题 / 双边全对 | 5/10 · 2/5 | 6/10 · 2/5 |
| 约 1,400 词元成对题，单题 / 双边全对 | 2/10 · 0/5 | 2/10 · 1/5 |
| 原长更正题，约 700 词元 | 5/10 | 7/10 |
| 原长更正题，约 1,400 词元 | 3/10 | 2/10 |
| 开放题规则通过，整组 / 逐轮 | 2/6 · 9/18 | 2/6 · 9/18 |
| 受控持久读取 | 32/32 | 32/32 |

最长的成对题中，一组做到了两边随新值变化，单题总分持平。原长更正题的两种长度表现方向相反。开放题沿用既有关键词规则，并有已知误判。保留训练结果用于对照，现有检查点继续作为起点。成对训练结果和权重保存在 `runs/spora-gelu24-recurrent-gated-counterfactual-v1/`。

## 单独训练定向擦除

在同一组约 1,400 词元原题上，完整注意力父模型回答新值 4 次、旧值 3 次。训练前的门控循环版回答新值 3 次、旧值 2 次，十题都能正常结束。下一次评测应分别检查旧值出现、新值缺失和未更改字段遗漏。

Gated DeltaNet-2 分开控制旧关联的定向擦除和新值写入。这里保留已有的按块全局衰减，在每个注意力头新增一项对擦除系数的独立修正，仍使用已归一化的块内键。新增权重初始化为零，起点在浮点舍入误差范围内复现原门控规则。单元测试验证了擦除强度变化只作用于测试中的匹配键，空状态下的新写入数值一致，分块与增量计算都能继续运行。这是该论文思想在现有标量门控上的块级近似。 [Gated DeltaNet-2 原文](https://arxiv.org/html/2605.22791v1#S3.SS1)

匹配预算对照从同一门控候选出发，两个条件各更新 64 步，再按同样的更正及持久读取指标评分。

```powershell
$sporaParent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaFirst = 'runs/spora-gelu24-recurrent-gated-delta-v4/layer-19'
$sporaSecond = 'runs/spora-gelu24-recurrent-gated-delta-v4/layer-22'
$sporaReference = 'runs/spora-gelu24-recurrent-gated-delta-v4-reference.json'
$sporaPairOutput = 'runs/spora-gelu24-recurrent-gated-counterfactual-v1'
.\.venv\Scripts\python.exe -m scripts.benchmark_spora_recurrent_dialogue $sporaParent $sporaReference $sporaFirst $sporaSecond --dialogue-only --extend
.\.venv\Scripts\python.exe -m scripts.train_spora_recurrent_counterfactual $sporaParent $sporaReference $sporaPairOutput $sporaFirst $sporaSecond --steps 32
foreach ($sporaRule in 'split_gated_block_delta','gated_block_delta') {
    $sporaOutput = "runs/spora-recurrent-$sporaRule-warm-v1"
    .\.venv\Scripts\python.exe -m scripts.train_spora_recurrent_long_alignment $sporaParent $sporaOutput $sporaFirst $sporaSecond --steps 64 --update-rule $sporaRule --history-projection unrotated
    if ($LASTEXITCODE) { break }
}
```
