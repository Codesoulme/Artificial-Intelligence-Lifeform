# 分开校准直接指令与相对答复

上一轮增加说话人对照后，模型在转述题上的读回有所改善，明确指令却更容易被搁置。本轮对两种读出使用相同的 31 种训练答复和 18 种校准答复。数据加入成对的直接继续与取消指令。两组都使用同一孢子候选的前 12 层特征及已保存的待确认动作。统一读出判断五种答复，分开读出分别判断直接动作及同意或反对。每个读出也可以返回待确认。只有一个读出超过其阈值并给出动作时才执行，否则保留待确认。 [PassiveQA](https://arxiv.org/abs/2604.04565) · [说话人对照](spora-lookup-speaker-confirmation-20261126.md)

| 留出指标 | 统一读出 | 分开读出 |
| --- | ---: | ---: |
| 原题，应读时实际读对 | 36/96 | 36/96 |
| 上一批新措辞，应读时实际读对 | 60/96 | 36/96 |
| 转述题，应读时实际读对 | 24/60 | 30/60 |
| 新混合题，应读时实际读对 | 30/48 | 24/48 |
| 新混合题，取消后仍读取 | 12/96 | 0/96 |
| 四套题，明确指定动作判对 | 12/44 | 0/44 |
| 四套题，不应读取时实际读取 | 24/1236 | 30/1236 |

统一读出的阈值为 0.99。分开方案为直接动作选择了 1.01，实际关闭了该读出，相对答复阈值为 0.65。直接动作读出把校准句「请停办手里的整批查询」判为继续，置信度约为 0.9996。这里的误判发生在直接继续和取消之间。现有校准目标通过关闭读出来避免它，代价是漏掉明确指令。新混合题中，分开方案虽没有在取消后读取，仍把一条转述型答复误判为取消，且取消后从非空范围清空并结束等待的次数为 36/72。

四套评测分别有 40、40、28 和 20 个问答组合，每个组合跨三组档案地址和四种初始范围。共 1536 次回放，包含 300 次应读检查。地址、版次和待确认动作由实验提供。当前结果显示，校准无法弥补明确动作之间的高置信混淆。两种读出继续作为离线实验。下一轮应直接对照仅改变行动动词或否定范围的明确指令，并把是否判对动作与是否应该执行分开测。

运行约 118 秒，受检权重与档案摘要一致，35 项相关测试通过。逐题结果及权重在 `runs/spora-lookup-split-response-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_confirmation runs/spora-lookup-transitions-clauses-v1 runs/spora-lookup-reset-matched-v1 runs/spora-lookup-split-response-new --recurrent @weights --stored-action --split-response
```
