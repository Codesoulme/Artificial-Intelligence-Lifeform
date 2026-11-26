# 分开识别查阅意图与地址

前一轮联合排序经常把无合适地址排在第一。这轮在同一冻结骨干上训练两个独立的头。一个判断查阅意图，另一个在人名和字段组合中预测地址。地址头在所有出现姓名的字段组合中先选首选，再核对这个地址是否在目录里。这样可以检验模型学到了哪个字段，并避免因首选缺失而重新排序。训练沿用 96 条请求，26 条题独立校准。分项指标借鉴了 PassiveQA 对追问与弃权的区分，以及 Xing USP 对字段和实例归属的单独评价。 [上一轮报告](spora-switch-address-ranking-20261126.md) · [PassiveQA](https://arxiv.org/abs/2604.04565) · [Xing USP](https://arxiv.org/abs/2610.34343)

| 正例 | 数量 | 首选人物正确 | 首选字段正确 | 通过校准和目录后提议正确 |
| --- | ---: | ---: | ---: | ---: |
| Windows 旧失败 | 3 | 3 | 3 | 2 |
| 新写口语问法 | 6 | 6 | 2 | 0 |
| 相似姓名 | 4 | 4 | 3 | 1 |
| 第二批独立措辞 | 4 | 4 | 2 | 1 |

独立校准选出意图阈值 0.99、地址阈值 0.5。校准集有 16 条正例，14 条的首选地址正确，阈值和目录检查后剩 8 条正确提议。新措辞里，五个首选地址正确的请求被意图阈值拒绝。人物均选对，错误主要在字段。第二批独立措辞有一条把书目理解成饮品，错误首选通过了校准和目录检查。另有一条询问未登记的书目，模型错误提议了该人物的用品。

对 17 条正例逐条移除正确地址后，四次正确提议全部被目录拦下。那条把书目理解成饮品的错误仍然存在，因为饮品从开始就是模型的首选。这显示了目录检查的覆盖边界。原口语头在相同留出上提议正确 3 次，且无错提议。分离组提议正确 4 次，并出现上述两类错误，继续留在实验入口。实验只提议地址，未执行持久记录读取。

实验约 108 秒，档案和源权重摘要一致。下一轮优先改字段区分，例如将同一人物的书目、用品、饮品做成成对样本。字段准确性和意图召回应继续分开衡量。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_switch_address_ranking runs/spora-lookup-transitions-clauses-v1 runs/spora-spoken-query-data-v2 runs/spora-dense-joint-address-v1 runs/spora-switch-address-separated-new --recurrent @weights --retrain --separate
```

逐题结果和实验权重位于 `runs/spora-switch-address-separated-v1`。
