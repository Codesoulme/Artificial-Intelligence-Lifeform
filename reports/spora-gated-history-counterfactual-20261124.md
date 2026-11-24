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

| 指标 | 续训前 | 原门控续训 | 独立擦除门续训 |
| --- | ---: | ---: | ---: |
| 约 700 词元更正，完整 | 5/10 | 5/10 | 5/10 |
| 约 1,400 词元更正，完整 | 3/10 | 4/10 | 4/10 |
| 约 1,400 词元，新值出现 | 3/10 | 4/10 | 4/10 |
| 约 1,400 词元，旧值出现 | 2/10 | 2/10 | 2/10 |
| 约 1,400 词元，保留字段出现 | 6/10 | 8/10 | 8/10 |
| 开放题规则通过，整组 / 逐轮 | 2/6 · 9/18 | 2/6 · 9/18 | 2/6 · 9/18 |
| 受控持久读取 | 32/32 | 32/32 | 32/32 |

两个续训组使用同一份数据、检查点、个体及长度分布。30 条更正和 18 轮开放题的可见文本逐条一致，32 条持久读取的完整记录也一致。最长组的改善在原门控续训中已经出现。此结果只覆盖当前的块级近似、训练预算和注意力对齐目标，现阶段保留原门控作为结构基线。

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

## 更正值在输出层的竞争

对 10 组成对问题重新打分。每对仅改动更正消息中的新值。比较两个新值的第一个分歧词元，如果值有共同开头，就给两边相同的开头。记录目标与另一个值的对数分数差，以及目标在完整词表中的排名。约 700 和 1,400 词元各五对。这一诊断使用给定的答值开头，自由生成另行评分。

| 注意力条件 | 约 700 词元，两边均选对 / 目标进前十 | 约 1,400 词元，两边均选对 / 目标进前十 |
| --- | ---: | ---: |
| 完整注意力父模型 | 4/5 · 5/10 | 4/5 · 5/10 |
| 独立擦除门，正常读远距状态 | 4/5 · 4/10 | 3/5 · 4/10 |
| 同一候选，关闭两处远距状态读出 | 3/5 · 6/10 | 4/5 · 6/10 |

最长组在打开远距状态时，8/10 次目标分歧词元胜过替代值，其中 4/10 进入完整词表前十。关读后分别为 9/10 和 6/10。最长的配送例中，关闭远距读出把两个目标的排名从 15、112 提到 3、6。物品例在三种设置下仍有一侧的目标排在替代值之后。其他解码层保留完整历史，所以关读对照只说明两处循环分支在本批输入上的边际作用。

另开进程重载同一候选，复现十条原最长更正题的完整答复。关闭两层远距读出后，完整通过从 4/10 降至 2/10，新值出现从 4/10 降至 2/10，旧值出现从 2/10 升至 4/10，保留字段从 8/10 降至 7/10。两组均正常结束。关读后两道样本寄送题只列字段名称，另有一道志愿者题答回旧组名。第一道配送题则补全了「捆」这个单位。词元排序和自由生成评价采用不同的答复条件，在本批数据上给出了方向不同的结论。

逐题分数和完整输出存于 `runs/spora-recurrent-split_gated_block_delta-warm-v1/` 下的 `preferences.json` 和 `far-generation.json`。

```powershell
$sporaSplit = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_value_preferences $sporaParent "$sporaSplit/preferences.json" "$sporaSplit/layer-19" "$sporaSplit/layer-22"
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_value_preferences $sporaParent "$sporaSplit/far-generation.json" "$sporaSplit/layer-19" "$sporaSplit/layer-22" --generate-extended
```

## 在哪个生成阶段读远距状态

固定两个替换层及其权重，仅改变它们在哪个阶段读取远距矩阵。前文预填充计算首个回答词元的分数。之后使用缓存逐词续写。四种设置都继续更新循环矩阵，因而逐词阶段仍可读取预填充时写入的内容。

| 启用读出的阶段 | 完整通过 | 新值出现 | 旧值出现 | 保留字段出现 | 正常结束 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 前文预填充和逐词续写 | 4/10 | 4/10 | 2/10 | 8/10 | 10/10 |
| 只在前文预填充 | 2/10 | 3/10 | 4/10 | 6/10 | 8/10 |
| 只在逐词续写 | 5/10 | 5/10 | 3/10 | 7/10 | 10/10 |
| 全程关闭读出 | 2/10 | 2/10 | 4/10 | 7/10 | 10/10 |

只在逐词续写读出的版本补全了两条配送量的单位，找回一条取展板地点和联系人，同时丢失两条样本寄送日期和收货人，因此净多完成一题。旧值出现数也上升。这十题用于找现象，分数倾向需通过新场景验证。逐题输出位于 `far-phase.json`。

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_value_preferences $sporaParent "$sporaSplit/far-phase.json" "$sporaSplit/layer-19" "$sporaSplit/layer-22" --phase-ablation
```

## 新场景复核

另写六种场景，每种两套值，在约 700 和 1,400 词元各生成 12 条完整答复。用之前的干扰段落和相同解码设置。三个条件使用逐字一致的输入和评分规则。复诊、取件、库存、节目单、花圃和放映场景的用户问句与原更正题无逐字重合；这些场景未用于训练或选择门控参数。 [新场景](../data/recurrent-phase-transfer-v1.json)

| 条件 | 约 700 词元，完整 / 新值 / 旧值 / 保留字段 | 约 1,400 词元，完整 / 新值 / 旧值 / 保留字段 |
| --- | ---: | ---: |
| 完整注意力父模型 | 1 / 4 / 3 / 7 | 1 / 4 / 3 / 6 |
| 两个阶段都读远距矩阵 | 1 / 3 / 5 / 7 | 1 / 4 / 2 / 5 |
| 只在逐词续写读远距矩阵 | 1 / 5 / 4 / 7 | 1 / 4 / 4 / 7 |

各项分母为 12。三种条件均只有同一条取件场景完整通过，72 条答复都正常结束。在逐词续写才启用远距矩阵，长场景多保留了两个字段，也多提到两个旧值。早先十题的净增加未在这批数据中复现。两个循环层的原读出时机继续作为结构基线。

两层循环版在最长库存题能报出「九只」，却省略接收人。节目单的两道题则只答出领唱人。这些现象与完全复用旧值不同。下一轮可在同一历史上分别问两个字段，再问合并问题。若单字段已能取出正确值而合并回答丢项，应把监督移到完整回答；若在原版完整注意力中就取不出值，先检查题目和语言骨干的能力。若主要问题仅出现在循环层，可用逐注意力头消融找出需要保留精确读取的位置。 [DuoAttention](https://arxiv.org/abs/2410.10819)

逐题输出在 `far-transfer.json` 和 `far-transfer-parent.json`，下方命令在一次运行中复现三个条件。

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_value_preferences $sporaParent "$sporaSplit/far-transfer-reproduce.json" "$sporaSplit/layer-19" "$sporaSplit/layer-22" --transfer data/recurrent-phase-transfer-v1.json
```
