# 口语改查的地址排序对照

Windows 回放中的三个口语请求均未得到具体地址提议。本轮复用现有的两个联合排序头，比较仅使用整句表示与结合人物附近表示的结果，并重训同一局部结构。排序候选只来自目录中已登记、且姓名在请求里出现的地址。模型也可以选择没有合适地址。D3ST 用描述匹配字段，Xing USP 将字段匹配与实例归属分开计分。本轮以姓名和字段同时匹配为准。 [入口回放](spora-spoken-switch-cli-20261126.md) · [D3ST](https://arxiv.org/abs/2201.08904) · [Xing USP](https://arxiv.org/abs/2610.34343)

现用口语头和两个已保存的联合头沿用各自原有阈值。局部重训组使用 96 条自编请求、26 条独立校准题。训练包括单人查阅、双人改查、取消和改写。对每条目标存在的训练请求再生成一份将正确地址移出目录的视图，将其标为无合适地址。校准同时对错提议与目标缺失时的替代提议扣分。四组都使用相同的入口资格过滤。

| 正确地址提议 | 现用口语头 | 整句联合头 | 已保存局部头 | 局部重训 |
| --- | ---: | ---: | ---: | ---: |
| Windows 旧失败 | 0/3 | 0/3 | 0/3 | 2/3 |
| 新写口语问法 | 2/6 | 0/6 | 0/6 | 0/6 |
| 相似姓名 | 1/4 | 0/4 | 0/4 | 1/4 |
| 第二批独立措辞 | 0/4 | 0/4 | 0/4 | 0/4 |

四组在 11 条无需提议的请求上均未给地址，将 17 条正例的正确地址逐条移除后也均未改选其他记录。相似姓名组沿用旧联合头的评估姓名，测试问法重新编写。三版档案只提供目录，实验未执行读回。

重训组恢复了「碰头」与「拿的东西」两条旧失败，在两批新措辞上仍全部将无合适地址排在第一。26 条校准题中也没有正确提议，最终阈值为 0。降阈值无法改变这些题的排序。候选留在实验入口。下一轮可在同一冻结表征上单独评估候选字段排序，并用独立训练的查阅意图判断承接缺失地址对照，检验字段识别收益是否伴随更多错误提议。

两次运行分别约 114 和 113 秒，实验前后档案及输入权重摘要一致。逐题结果和重训权重位于 `runs/spora-switch-address-ranking-v2`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_switch_address_ranking runs/spora-lookup-transitions-clauses-v1 runs/spora-spoken-query-data-v2 runs/spora-dense-joint-address-v1 runs/spora-switch-address-ranking-new --recurrent @weights --retrain
```
