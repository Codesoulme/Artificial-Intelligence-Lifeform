# 较早信息经过孢子循环状态的对照

四组布局比较后，第 19、22 层继续作为结构基线。它在约 1,400 词元时只完整答对 2/10，仍缺少处理多条有效要求的能力。近期研究发现，多种混合模型倾向使用精确注意力，普通监督微调仍可能加重这种依赖。研究者在额外的训练前向中阻止精确注意力跨段访问，再保留常规目标，Qwen3.5-4B 和 Nemotron-H-4B 在六类问答上的平均准确率分别增加 3.9 和 5.2 个百分点。孢子当前只迁移了两层，因此先在本机检验读出路径，再选择是否做辅助训练。 [层组合结果](spora-hybrid-layer-placement-20261125.md) · [Balancing Memory Pathways](https://arxiv.org/abs/2610.06750)

## 对照

复用最长的 10 条更正题和相同的第 19、22 层权重。保留词元次序和原有绝对位置，从第 512 个位置隔开两段。登记、更正以及需要保留的事实都早于分段点，分段点后的词元中也逐题检验了无待评分值的副本。常规生成保持无截断的精确注意力。受限条件下，普通注意力层只能在自己的段内因果访问，循环层继续处理完整序列。

同一批输入比较三种受限条件。完整注意力父模型作为基准。两组循环层分别打开和关闭远距状态读取，两组的近期精确窗口相同。窗口可跨分段点，因此两组差异测的是远距读取在相同近期窗口上的贡献。计分分别记录更新值、应保留字段、已过期值以及完整答复。这轮只加载已保存的个体进行诊断。

CPU 检查覆盖了未来词元、跨段屏蔽、提示预填充与增量生成的末行一致性，用玩具注意力验证分段后的查询无法读到早段值。GPU 结果与未受限基准使用同一套提示、检查点和个体摘要。

## 结果

| 条件 | 更新值出现 | 应保留值出现 | 过期值出现 | 完整通过 |
| --- | ---: | ---: | ---: | ---: |
| 第 19、22 层循环版，无跨段限制 | 3/10 | 5/10 | 2/10 | 2/10 |
| 父模型，段内精确注意力 | 0/10 | 0/10 | 0/10 | 0/10 |
| 循环版，段内精确注意力，开远距读取 | 0/10 | 0/10 | 0/10 | 0/10 |
| 循环版，段内精确注意力，关远距读取 | 0/10 | 0/10 | 0/10 | 0/10 |

各组均在输出预算内正常结束。远距读取开关改变了受限条件下全部十条的生成文字，却没有恢复任何应答值。例如取展板题仍会要求用户提供地点和联络人。核对了结果中的原始历史、目标值和来源摘要，诊断用时约 159 秒。

## 等预算训练

把一部分长样本的训练改为段内精确注意力，给循环层学习传递更正值和仍有效字段的机会。每一对样本的分段点相同，放在最后一次有效更新之后，至少隔开两个近期块。分段点后须有至少 96 个提示词元，并逐例排除出现待答值副本的样本。沿用原来的样本顺序，每隔一个合格的长样本对给予分段约束，其余步采用常规训练。按已保存的基线抽样顺序预核对，32 次中有 16 次满足条件，安排 8 次分段训练。论文使用更大规模的混合骨干并合并两项损失。本地实验用轮替保持总前向次数一致。 [训练设计参考](https://arxiv.org/html/2610.06750v1)

对照使用同一第 19、22 层的位置、已有候选权重、父检查点、学习率及抽样顺序。恢复完分别评估常规对话、受限条件下开关远距读取，以及具名持久读取。受限对照会从相应训练报告加载普通答复，并核对生成报告和新层权重的摘要。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_recurrent_pathways runs/spora-composition-late-gates-20-23-24-v1/candidate.pt runs/spora-recurrent-dialogue-extended-v1.json runs/spora-recurrent-pathways-v1.json runs/spora-gelu24-recurrent-layer19-v2 runs/spora-gelu24-recurrent-layer22-v1
```

```powershell
$sporaParent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaReference = 'runs/spora-recurrent-dialogue-extended-v1.json'
$sporaControl = 'runs/spora-morph-recovery-19-22-v1'
$sporaAux = 'runs/spora-recurrent-segment-supervision-v1'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_recurrent_counterfactual $sporaParent $sporaReference $sporaAux runs/spora-gelu24-recurrent-layer19-v2 runs/spora-gelu24-recurrent-layer22-v1 --steps 32 --learning-rate .0005 --segment-supervision
foreach ($sporaRun in $sporaControl,$sporaAux) {
    .\.venv\Scripts\python.exe -u -m scripts.audit_spora_recurrent_pathways $sporaParent $sporaReference "$sporaRun/pathways.json" "$sporaRun/layer-19" "$sporaRun/layer-22" --recovery $sporaRun
    if ($LASTEXITCODE -ne 0) { break }
}
```
