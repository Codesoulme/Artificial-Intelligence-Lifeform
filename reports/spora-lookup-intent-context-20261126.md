# 整句信息能否帮助判断查阅范围

单独看前半句时，「请你说说某项」没有线索表明用户可能在后半句撤销它。借鉴 StateWise 对新增和撤销的成对评测，本轮在原有训练集上添加 16 条包含追加、替换或撤销的视图。两组都在双循环层后的同一位置提取特征，用相同的 160 条训练视图和同一个字段头。一组只看各分句，另一组额外接收整句末位的隐藏状态。后者输入维度更大，意图头的参数数目也增加。阈值按同一规则在 64 条校准题上选择，统计两个分句是否都判对，并额外扣除零项误判。 [StateWise](https://arxiv.org/abs/2610.33766) · [SAGE](https://arxiv.org/abs/2609.00434) · [同深度对照](spora-lookup-intent-attention-control-20261126.md)

| 留出指标 | 逐分句 | 增加整句信息 |
| --- | ---: | ---: |
| 分句各自说明意图时，单项正确提议 | 12/32 | 7/32 |
| 分句各自说明意图时，双项均识别 | 10/16 | 10/16 |
| 后句修正前句时，单项正确提议 | 2/8 | 2/8 |
| 后句修正前句时，双项均识别 | 2/4 | 4/4 |
| 后句取消全部时，误提议单项 | 4/4 | 4/4 |
| 原第一批留出单项正确提议 | 20/24 | 24/24 |
| 原暂缓与改写的多余提议 | 0/8 | 2/8 |

整句信息在「原先拜托你的照常，再加上……」这一措辞下改善了双项识别。四条测试只替换人物和事项，因此还需要独立的追加措辞确认。两组都没有学会「两件一律先搁置」的整体撤销。整句特征的普通单项成绩下降，新头均保留为离线权重。下一轮可直接对「保留、撤销、追加」的状态变化做监督，同时在实际连续对话中评估取消后是否仍提议旧记录。

初跑的校准明细将普通分句和后句修正的题组共用编号。改用独立编号后复跑，两组的所有汇总计数一致。正式运行约 156 秒，输入权重摘要一致，35 项相关测试通过。结果保存在 `runs/spora-lookup-intent-context-v2`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_intent_depth runs/spora-switch-address-matched-v1 runs/spora-lookup-field-focus-independent-v1 runs/spora-lookup-schema-ranking-v1 runs/spora-lookup-intent-context-new --recurrent @weights --contextual
```
