# 查档更新的分句对照与逐项动作

上一轮配对的留出结果是 0/48。这轮沿用 1,088 条训练样本、240 步及 360 条留出题，将整句表示与单独编码的分句均值拼接。原条件只用整句。两组共用整句组学得的版次头，动作头的输入宽度随拼接变化。每轮记录两个候选分别收到的动作和有效状态。TPBench 的对照表明，保住事实和保住改变事实的那句话需要分开测。 [上一轮报告](spora-lookup-transition-pairs-20261126.md) · [TPBench](https://arxiv.org/abs/2610.02736)

| 指标 | 整句 | 整句加分句 |
| --- | ---: | ---: |
| 360 条中正确决策 | 207 | 258 |
| 144 条应查请求实际读对编号与名称 | 45 | 60 |
| 216 条无需直接读取的请求被读取 | 54 | 18 |
| 地址正确、版次错误 | 9 | 12 |
| 新措辞配对两侧都对 | 0/48 | 0/48 |

48 对中有六组不同的开场及地址组合，其余变化是追问版次。两组在六个组合的追加、改查两侧，均对原地址输出保留，对新地址输出选入。整句组一半开场遗漏原地址，分句组在全部六组开场都正确选入原地址。手工给对开场地址后，两组在改查侧仍保留原地址，所以配对通过数仍为零。分句组的首次真实读回改善与开场识别同步，但本实验还改变了学习得到的置信度及独立校准阈值，不能把 15 条增益全部归因于开场。两套阈值分别为 `(0, 0)` 和 `(0.5, 0)`。

额外的 12 条版次问法对照没有向模型提供地址或事实。共用版次头在首版、当前版、上一版、非查阅措辞中分别答对 3/3、2/3、0/3、2/3。两条新的上一版问法被判成首版，另一条被判作普通对话。这解释了为何上一轮熟悉措辞的 18 次实际上一版读回成功，仍需要独立测试自然问法的迁移。 [TrackMem](https://arxiv.org/abs/2611.02001)

两组同进程约 146 秒。旧整句基线的 360 条决策逐条复现。123 项相关测试通过，重开的三版实验档案摘要一致。权重及逐项结果在 `runs/spora-lookup-transitions-clauses-v1`。候选仍在实验入口。接下来应把「保留或撤掉已确认项」单独作为操作预测，检查模型在不同否定、改口与引用结构下是否随原项指代改变动作。与版次头分别使用独立的措辞留出。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_transitions runs/spora-paired-lookup-clarification-typed-read-v1 runs/spora-lookup-transitions-clauses-new --recurrent @weights --paired-transitions --compare-clauses
```
