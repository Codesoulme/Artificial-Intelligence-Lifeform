# 孢子循环注意力的成对更正训练

从原第 19、22 层循环注意力权重出发。上一轮的长历史拟合改善了局部向量误差，但约 1,400 词元时仍只能完整回答 2/10。这次用最终答案监督循环层，冻结整个 DeepSeek 骨干和持久记忆个体。选择原双层权重作为起点，因为它在约 700 词元的既有对照中完整通过 7/10，长历史拟合版为 6/10。 [前一轮对齐](spora-recurrent-long-alignment-20261124.md)

每对训练输入保留相同的登记、追问、无关信息和它们的位置，只改变更正消息指出的新值。全序列损失包含所有可见答案词元与结束符。在两个答案第一次分歧处，另外要求正确词元分数高于另一份记录的词元。两个方向各算梯度，再集中更新。重计算后段解码层以减少反向传播时的显存占用。位置表临时扩至 2,048，原位置系数逐项校验。 [Focused Transformer](https://arxiv.org/abs/2307.03170)

评测在约 700 与 1,400 词元各用五对更正记录。每对两题必须回答各自的新值、保留同一未更改字段、排除旧值和另一题的新值，并正常结束，才算完整通过。另对原二十条更正和六组开放三轮题复测。开放组沿用原规则，存在已备案的关键词误判。 [评测规则](spora-recurrent-dialogue-cost-20261124.md)

首轮在训练后对照期间中断，当时权重尚未保存。复跑把训练权重和记录移到评测前保存。需要接续时可用 `--evaluate` 加载已保存的权重。训练组和起点组使用同一套成对题。

## 结果

从 48 个训练对抽样更新 32 步，实际使用 31 个不同的对。全程约 9 分钟，训练的 PyTorch 峰值分配为 4.07 GiB。两层各只更新循环路径的投影与混合门。

| 条件 | 约 700 词元，单题 / 成对 | 约 1,400 词元，单题 / 成对 | 原长更正，700 / 1,400 词元 | 开放题规则通过，组 / 轮 |
|---|---:|---:|---:|---:|
| 起点 | 5/10 · 2/5 | 3/10 · 1/5 | 7/10 · 2/10 | 2/6 · 8/18 |
| 成对训练后 | 5/10 · 2/5 | 3/10 · 1/5 | 7/10 · 2/10 | 2/6 · 8/18 |

成对评测共有一条答案文字变化，题目通过情况没有变化。约 1,400 词元时，两套包裹更正都能写出收件人与数量数字，仍丢掉「捆」或「册」。另有两个会议问句原样重复给用户。32 条受控持久记忆答复全部正确，个体状态散列一致。现有结果支持改动历史状态的写入规则，继续用相同指标比较简单累加和按键定向更新。已核对 Gated DeltaNet-2 摘要，其擦除门和写入门分离的做法可作为架构对照。 [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)

权重、训练记录和逐题结果位于 `runs/spora-gelu24-recurrent-counterfactual-v2/`。独立重载逐项复现 20 条成对题、20 条原更正题、18 轮开放题和 32 条持久记忆答复。29 项循环注意力、位置边界和持久化测试通过。

```powershell
$sporaBase = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaReference = 'runs/spora-recurrent-dialogue-extended-v1.json'
$sporaOutput = 'runs/spora-gelu24-recurrent-counterfactual-v3'
$sporaFirst = 'runs/spora-gelu24-recurrent-layer19-v2'
$sporaSecond = 'runs/spora-gelu24-recurrent-layer22-v1'
.\.venv\Scripts\python.exe -m scripts.train_spora_recurrent_counterfactual $sporaBase $sporaReference $sporaOutput $sporaFirst $sporaSecond
.\.venv\Scripts\python.exe -m scripts.train_spora_recurrent_counterfactual $sporaBase $sporaReference $sporaOutput $sporaFirst $sporaSecond --verify
```
