# 用成组措辞训练撤销与继续的区分

上一轮的三分类容易把「不得中止」和转述别人取消判成用户撤销。NevIR 用接近的措辞测试否定是否改变判断，SpeakerMem-R1 强调独立核对说话人归属。本轮在原有 44 句上增加 8 组、24 句训练数据，每组保留共同措辞，改变否定位置、发言来源或用户是否作出决定。两组三分类头使用相同的冻结孢子双循环层候选、优化配置和 12 句校准集，分别选出阈值。 [NevIR](https://arxiv.org/abs/2305.07614) · [SpeakerMem-R1](https://arxiv.org/abs/2609.26780) · [三分类基线](spora-lookup-reset-clarification-20261126.md)

| 指标 | 原训练集 | 增加成组措辞 |
| --- | ---: | ---: |
| 校准阈值 | 0.95 | 0.99 |
| 旧题意图正确 | 6/18 | 7/18 |
| 新题意图正确 | 9/18 | 9/18 |
| 两套题误取消 | 6/24 | 0/24 |
| 两套题明确指令却追问 | 15/24 | 20/24 |
| 新题整组三句均正确 | 1/6 | 0/6 |
| 回放应读时读对记录、版次和值 | 66/144 | 78/144 |
| 回放不应读取却读取 | 0/720 | 0/720 |
| 确认继续后的正确读回 | 54/72 | 72/72 |
| 确认取消后仍处于待确认 | 0/144 | 144/144 |

回放覆盖三组档案地址、四种初始查阅范围。确认继续和确认取消各使用一种措辞，待确认的两条分支会重复计入确认前检查。增强组识别了确认继续，但同一句明确确认取消在每次回放中都未结束待确认。暂停读取和清空查阅范围分别计分。候选仍在离线实验中。

用保存的分数在相同 0.95 阈值回看，增强组在 36 句中判对 22 句，原组判对 15 句。误取消从 6 句降到 0 句，多余追问从 15 句降到 13 句。这个回看使用了留出数据，不能用于选择发布阈值。下一轮应给待确认状态下的明确答复单独设校准和留出，覆盖多种取消措辞，并核对范围实际清空后再问「上一版」是否保持不读。

本轮约 121 秒，受检权重和档案摘要一致，35 项相关测试通过。逐句结果和保存的权重位于 `runs/spora-lookup-reset-matched-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_reset_clarification runs/spora-lookup-transitions-clauses-v1 runs/spora-lookup-global-reset-v1 runs/spora-lookup-reset-matched-new --recurrent @weights --matched
```
