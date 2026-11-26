# 撤销、继续与待确认的查阅回放

整体撤销的二分类读出会把「一个也别给我停掉」判成取消。本轮参考 PassiveQA，将继续、取消和待确认分别作为监督标签。使用同一孢子双循环层候选、候选动作头和三版档案。新的读出用 44 条句子训练，12 条独立句子校准。低于阈值的决定转入待确认。待确认时保留范围并暂停读取，明确继续后恢复，明确取消后清空。 [PassiveQA](https://arxiv.org/abs/2604.04565) · [StateWise](https://arxiv.org/abs/2610.33766) · [整体撤销基线](spora-lookup-global-reset-20261126.md)

留出集有六组独立措辞，每组各有明确取消、明确继续和待确认一句。评测从四种初始范围和三组档案地址出发。待确认题各有继续、取消两个分支。两个分支重复计入确认前检查，因此每种方案共有 432 次「上一版」读回检查。这是离线回放，候选地址和版次由评测提供。两路还使用了不同的训练数据，结果反映整套读出与决策方案的差异。

| 指标 | 二分类 | 三分类 |
| --- | ---: | ---: |
| 全部应读轮次中读对记录、版次和值 | 6/72 | 30/72 |
| 不应读取时实际读取 | 6/360 | 0/360 |
| 明确取消后范围为空（含初始为空） | 63/72 | 27/72 |
| 明确继续后漏读 | 30/36 | 30/36 |
| 待确认时正确进入等待 | 0/144 | 96/144 |
| 用户确认继续后读对上一版 | 0/36 | 24/36 |
| 用户确认取消后仍读取 | 0/72 | 0/72 |

三分类从校准选择 0.95。六句明确取消只有一句直接判成取消，其余五句进入待确认。「每一笔核验都不得中止」被误判为取消。六句待确认中有两句带有转述或引用，也被直接判成取消，导致后续确认继续时无法恢复原选择。这些失误使其继续留在实验中。下一轮可用成对反例训练说话人和否定作用范围，并在独立题上同时计直接取消、误取消和不必要的追问。

运行约 122 秒，输入权重与档案摘要一致，35 项相关测试通过。逐题结果和离线权重在 `runs/spora-lookup-reset-clarification-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_reset_clarification runs/spora-lookup-transitions-clauses-v1 runs/spora-lookup-global-reset-v1 runs/spora-lookup-reset-clarification-new --recurrent @weights
```
