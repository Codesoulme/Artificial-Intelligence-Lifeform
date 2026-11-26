# 分开学习查阅范围的更新和版次判断

上轮直接预测当前查阅范围，在双项和撤回场景中表现较好，但改查新地址为 0/48。StateWise 分开检查约束撤销和后续执行，TIDE-Bench 把改口识别及恢复任务分开计分。此次让模型逐条预测保留、选入或移出候选地址，并单独训练末问的版次判断。 [上轮报告](spora-recurrent-lookup-scope-20261126.md) · [StateWise](https://arxiv.org/abs/2610.33766) · [TIDE-Bench](https://aclanthology.org/2026.emnlp-main.2454/)

固定原两层循环骨干、档案、划分和 240 步训练预算。将文字中完整出现的两个候选地址分别替换为「本项记录」和「另一项记录」，从冻结 DeepSeek 提取每个候选的视角。更新头预测动作，只把结果写入当前候选的有效位。末问用独立的小头预测首版、当前版、上一版或普通聊天。只有预测恰有一个有效地址时才提议读取。读取置信度和版次置信度单独校准。对照另用人工标注的地址范围，只用于测剩余版次及意图误差。

首轮沿用先前的特征提取方式，在标记后的文字末尾再附候选说明。第二轮只编码标记后的消息，分别取末尾六字符的隐藏表示。版次头也改用直接编码的末问。两轮的训练数据、种子和分类结构相同，两个头的输入特征都随实验条件改变。测试继续使用 264 条留出，其中 108 条应查。地址和值只在预测完毕后用于实际张量读回。

| 指标 | 上轮循环头 | 更新动作，附候选说明 | 更新动作，直接编码 |
| --- | ---: | ---: | ---: |
| 起始地址正确，排除无历史项 | — | 120/240 | 120/240 |
| 更新后范围正确，改查 | — | 0/48 | 24/48 |
| 更新后范围正确，保留两项 | — | 0/48 | 0/48 |
| 校准后整项决策正确 | 126/264 | 122/264 | 183/264 |
| 校准后读对记录和值 | 0/108 | 17/108 | 45/108 |
| 校准后应查题选错地址 | 0 | 12 | 0 |
| 校准后地址正确但选错版次 | 0 | 7 | 9 |
| 校准后无需读取却读取 | 0/156 | 27/156 | 18/156 |
| 给定正确范围时，选对版次及读取动作 | — | 78/108 | 90/108 |

直接编码的 45 次正确读回，保持原项、改查和仅提及另一项各占 15 次。18 次不该读取却读取全部发生在追加第二项时。给定正确起始地址的对照中，保持原项和仅提及另一项均从 24/48 升到 48/48，改查仍为 24/48，保留两项仍为 0/48，撤回仍为 24/48。起始识别和后续操作都有失误。直接编码组在独立校准中选出的两个阈值均为 0。未见措辞仍产生误读，当前结果保留在实验入口。

两轮分别用时约 165 和 117 秒，存档和源权重摘要一致，121 项相关单元测试通过。同一场景有多个末问，表内不同问题可能共用同一个范围更新，因此 48 条改查结果对应六组地址与表述的组合。下一轮优先加入配对的追加与改查句，要求实际同一候选在两种命令里分别保留和移出。版次另用同一地址三份以上记录核对，以区分首版和上一版。TrackMem 的修订题也按有效值和过期值分开评价。 [TrackMem](https://arxiv.org/abs/2611.02001)

## 复现

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_transitions runs/spora-paired-lookup-clarification-typed-read-v1 runs/spora-lookup-transitions-new --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_transitions runs/spora-paired-lookup-clarification-typed-read-v1 runs/spora-lookup-transitions-utterance-new --recurrent @weights --utterance-features
```

逐轮结果和小头权重在 `runs/spora-lookup-transitions-v1` 与 `runs/spora-lookup-transitions-utterance-v1`。
