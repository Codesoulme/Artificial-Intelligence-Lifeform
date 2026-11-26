# 双循环版的中文事实更正恢复

此前在三个场景中，重复字段名能改善远距更正，但并未让模型稳定复述仍有效的另一个值。本轮参考对话状态研究对字段、取值和实际答复的拆分，以及 StateMemBench 对当前值和已作废值的分开评分。训练同时监督更新值和保留值。留出评测要求答对所问字段、正常结束，并检查是否复述旧值。 [字段指代对照](spora-hybrid-revision-field-reference-20261127.md) · [Lost with a Map](https://arxiv.org/abs/2609.33883) · [StateMemBench](https://arxiv.org/abs/2608.19652) · [RECON](https://arxiv.org/abs/2607.16716)

两组从同一份第 19、22 层循环权重起步，各更新 32 次，只训练循环核。对照使用 32 条 CrossWOZ。混合组交错使用前 16 条 CrossWOZ 和 16 条自编更正。两组使用相同学习率和答复训练目标。更正数据来自八个新场景，有八条双字段、四条更新值单字段和四条保留值单字段。每种目标均覆盖近距、远距、重复字段名和省略指代。更正训练提示长 440 至 458 词元，留出生成提示约 700 词元。留出题的场景名、问题和事实值均与自编训练集无交集。

| 指标 | 起点 | 仅 CrossWOZ | CrossWOZ 加更正 |
| --- | ---: | ---: | ---: |
| CrossWOZ 留出 NLL，48 段 | 4.94577 | 4.92201 | 4.93193 |
| 近距单字段 | 8/10 | 8/10 | 8/10 |
| 远距单字段 | 3/10 | 3/10 | 3/10 |
| 近距双字段 | 1/10 | 1/10 | 1/10 |
| 远距双字段 | 0/10 | 0/10 | 0/10 |
| 远距双字段答复包含新值 | 0/10 | 0/10 | 0/10 |
| 远距双字段答复包含保留值 | 5/10 | 5/10 | 5/10 |
| 远距双字段答复复述旧值 | 6/10 | 6/10 | 6/10 |
| 具名持久读取 | 32/32 | 32/32 | 32/32 |

两组各 40 条生成均正常结束。两组之间有 5 条答复文本不同，40 道题的逐题通过判定相同。训练步数相同，数据量有差异。混合组用了 9,130 个提示词元和 673 个监督词元，对照用了 3,768 个提示词元和 870 个监督词元。两组分别用时约 231 和 224 秒，训练张量峰值均为 3.65 GiB。独立重载与生成分别约 219 和 192 秒。训练和评测均核对源权重和个体状态摘要，26 项相关回归通过。产物在 `runs/spora-revision-mixed-recovery-v1` 和 `runs/spora-revision-replay-recovery-v1`。

32 步混合训练在这批留出场景没有带来可测的答复提升。实验没有扩大到另一套生成题，也没有复测开放聊天。原双层版继续作为本机候选。下一轮先在双层版里检验从长对话提取两个字段的有效版本，并对照已知正确地址和模型推断地址。读取时复用已有持久张量，分别测地址选错、取值错误和答复漏值。此前把两个正确值直接放入提问后，双层版仍只答全 14/20，因此地址准确和生成完整都要单独验收。 [给值对照](spora-hybrid-revision-stages-20261127.md) · [持久读出](spora-persistent-value-reader-20261126.md)

## 复现

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$base = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$train = 'runs/research/crosswoz-df82c9f/train.json.zip'
$test = 'runs/research/crosswoz-df82c9f/test.json.zip'
$layers = @('runs/spora-hybrid-position-recovery-v1/19-22/layer-19', 'runs/spora-hybrid-position-recovery-v1/19-22/layer-22')
$mixed = 'runs/spora-revision-mixed-recovery-new'
$replay = 'runs/spora-revision-replay-recovery-new'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_hybrid_language_recovery $base $train $test $mixed @layers --layout '19,22' --steps 32 --revision-train data/recurrent-revision-recovery-20261127.json
.\.venv\Scripts\python.exe -u -m scripts.train_spora_hybrid_language_recovery $base $train $test $replay @layers --layout '19,22' --steps 32
foreach ($candidate in @($mixed, $replay)) {
    .\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_recovery $base $candidate "$candidate/distance.json" --skip-parent --revision-distance --recall-transfer data/recurrent-layout-recall-transfer-20261126.json --layout '19,22'
}
```
