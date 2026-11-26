# 重复字段名与省略指代的更正对照

图书场景的分句重排改善了存放处的回答，新标签仍未恢复。本轮换用展箱清点、织布体验和市集海报三个新场景。更正消息的首句始终指出要调整的字段，第二句分别用「这项」或重复字段名。其余消息、取值和追问相同。每个场景有两套取值，交叉测试近距、远距以及更新和保留字段。完整注意力父检查点、双循环版和三循环版分别生成 48 条答复。 [分句顺序对照](spora-hybrid-revision-field-order-20261127.md) · [D3ST](https://arxiv.org/abs/2201.08904) · [Lost with a Map](https://arxiv.org/abs/2609.33883)

| 布局 | 远距更新值，指代 → 重复字段名 | 近距保留值 | 远距保留值 | 远距两个字段都答对 |
| --- | ---: | ---: | ---: | ---: |
| 完整注意力父检查点 | 4/6 → 5/6 | 1/6 → 1/6 | 2/6 → 2/6 | 2/6 → 2/6 |
| 第 19、22 层循环版 | 4/6 → 6/6 | 2/6 → 1/6 | 4/6 → 4/6 | 2/6 → 4/6 |
| 第 13、19、22 层循环版 | 3/6 → 6/6 | 3/6 → 2/6 | 2/6 → 3/6 | 1/6 → 3/6 |

三种布局在近距更新值上均为 6/6。两种循环版的近距保留字段各少答对一题。双循环版的远距保留值总分持平，其中一题改善、一题退步。三循环版在使用指代时有一次复述完整旧值，重复字段名后未再复述。144 条生成均正常结束。

重复字段名会使输入增加 1–3 个词元。各对照沿用冻结权重，数据只覆盖三个任务。结果支持在下一轮恢复训练中同时保留两种表达，并在独立场景分别验收更新值、保留值及两个字段同时答对的结果。三循环版在明确重复字段名的条件下恢复了这批远距更新题，保留字段仍弱于双循环版。

实验约 358 秒，受检文件与个体状态摘要一致，35 项相关测试通过。复现数据位于 `data/recurrent-field-reference-20261127.json`，逐题输出位于 `runs/spora-hybrid-revision-field-reference-v1.json`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_recovery runs/spora-composition-late-gates-20-23-24-v1/candidate.pt runs/spora-hybrid-position-recovery-v1 runs/spora-hybrid-revision-field-reference-new.json --revision-field-reference --recall-transfer data/recurrent-field-reference-20261127.json --layout '19,22' --layout '13,19,22'
```
