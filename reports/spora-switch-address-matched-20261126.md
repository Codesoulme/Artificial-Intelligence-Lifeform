# 同一人物的四字段查阅对照

将饮品、见面地点、用品和书目放进同一句式，用同一人物及语气询问四种记录。字段头在原有 96 条请求上增加了 40 条请求。两组使用相同冻结骨干、初始化、48 次参数更新、意图头及校准阈值。增强组的每批样本构成随数据变化。新评测用两位未参与此次训练的人物和三组新措辞。每个人的四个字段都在候选目录里。整组通过要求按首选地址答对四项。 [上一轮结果](spora-switch-address-separated-20261126.md) · [SGD-X](https://arxiv.org/abs/2110.06800) · [Xing USP](https://arxiv.org/abs/2610.34343)

| 指标 | 原字段头 | 增加成组问法 |
| --- | ---: | ---: |
| 原四套留出首选地址正确 | 10/17 | 16/17 |
| 原四套留出通过校准和目录的正确提议 | 4/17 | 7/17 |
| 新四字段对照首选地址正确 | 18/24 | 18/24 |
| 新四字段对照整组通过 | 2/6 | 0/6 |
| 新四字段对照正确提议 | 10/24 | 6/24 |
| 新四字段对照错提议 | 0 | 4 |

新四字段对照中，增强组的地点从 4/6 提升到 6/6，饮品从 6/6 降至 4/6，两组的书目和用品分别为 2/6 和 6/6。同一个意图头在新对照中放行 10 条。原字段头全部选对，增强组有四条选错仍通过了目录检查。移除这四条的正确地址后，错误首选仍在目录中，因而继续通过。原四套留出上，增强组没有出现错误或多余提议。

这批新增样本带来的收益依赖具体措辞，字段头继续留在实验目录。实验使用目录元数据评估提议，未读出档案值。下一轮优先测书目与用品，并在两个字段的问法中加入同样的干扰线索，检验模型是否能根据真正受询问的对象稳定选项。

整轮约 99 秒，档案及源权重的摘要在前后一致。126 项相关测试通过。逐题结果与权重位于 `runs/spora-switch-address-matched-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_switch_address_ranking runs/spora-lookup-transitions-clauses-v1 runs/spora-spoken-query-data-v2 runs/spora-dense-joint-address-v1 runs/spora-switch-address-matched-new --recurrent @weights --separate --matched
```
