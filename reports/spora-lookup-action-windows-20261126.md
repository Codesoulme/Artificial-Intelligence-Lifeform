# 局部特征能否改善查阅指令识别

此前直接动作读出把「请停办手里的整批查询」高置信判为继续。借鉴 NevIR 的最小对照，本轮固定孢子候选、前 12 层、训练集和分类头结构，比较两种特征。基线使用整句均值与各分句均值的汇总。局部组保留整句均值，第二部分改为对所有四字符窗口的均值逐维取最大值。窗口选择不需要动作词标注。 [NevIR](https://arxiv.org/abs/2305.07614) · [上轮对照](spora-lookup-split-response-20261126.md)

两组使用 38 条训练答复。校准使用 18 条独立答复，同一规则给出各组阈值。留出包含动作词、否定范围和转述三类各四对，以及四条尚未决定的答复。每句从三组已知档案地址和四种初始范围回放「上一版」，以实际记录编号、版次和值计分。

| 留出指标 | 整句与分句均值 | 增加局部窗口 |
| --- | ---: | ---: |
| 校准阈值 | 0.80 | 0.90 |
| 24 条明确指令判对 | 14/24 | 13/24 |
| 12 对相反指令两边均判对 | 3/12 | 2/12 |
| 明确指令转入待确认 | 3/24 | 6/24 |
| 尚未决定时保留待确认 | 4/4 | 4/4 |
| 应读时实际读对 | 48/72 | 42/72 |
| 应取消时仍读出旧记录 | 24/144 | 18/144 |
| 从非空范围开始，取消后清空且结束等待 | 54/108 | 54/108 |

两组都正确识别「停办」校准句，但动作词组中没有一对在任一方案中全部判对。固定阈值 0.90 事后复算，基线与局部组分别判对 14 条和 13 条明确指令，各执行了 6 条和 5 条错误指令，其余错误进入待确认。两组阈值不同带来的回放差异不能全部归于特征。两组均留在离线实验中。持续提升的下一步是用冻结的同一候选比较现有分类头和原底座的受约束动作判断，同时把实际交流中的误读、漏读和追问一起核对。

运行约 112 秒，受检权重和档案摘要一致，35 项相关测试通过。逐题结果及权重位于 `runs/spora-lookup-action-windows-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_action_windows runs/spora-lookup-transitions-clauses-v1 runs/spora-lookup-action-windows-new --recurrent @weights
```
