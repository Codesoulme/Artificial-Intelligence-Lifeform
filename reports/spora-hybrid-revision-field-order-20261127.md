# 图书场景的分句顺序对照

在图书场景中分别调换初始登记和更正消息内的两个分句。每次保留原分句文字，把「手边的标签不够了，改用……」作为一块移动，两条消息在历史里的位置、取值和提问保持原样。对照包含两套取值、近距和远距追问，每种布局各生成 32 条答复。程序逐题校验同一问法在四种顺序下的提示词元数相同。 [上一轮字段切换](spora-hybrid-revision-field-switch-20261127.md) · [Lost in the Middle](https://arxiv.org/abs/2307.03172)

两种骨干在以下指标上相同。

| 顺序 | 新标签精确匹配（近距 / 远距） | 存放处精确匹配（近距 / 远距） |
| --- | ---: | ---: |
| 原顺序 | 0/2 · 0/2 | 0/2 · 2/2 |
| 只调换初始登记 | 0/2 · 0/2 | 0/2 · 2/2 |
| 只调换更正消息 | 0/2 · 0/2 | 2/2 · 2/2 |
| 两条消息均调换 | 0/2 · 0/2 | 2/2 · 2/2 |

在原顺序的近距场景，询问存放处得到「原处」或字段名。将「未登记的书还放在原处」移到更正消息开头后，两组都答出了最初登记的实际位置。新标签仍答成旧值、残缺值或其他文本。其中只调换初始登记时，双循环版远距两条都报了完整旧值，父检查点只有一条。64 条答复均正常结束。

此轮只对一个场景做了分句重排。移动更正分句也改变了答复前最后出现的话题，无法单独区分词元位置和话题顺序的影响。现有结果表明单靠重排没有恢复新标签的生成。下一轮可直接对照明确重复字段名的更正形式与沿用省略的形式，并用新场景检验是否值得把这种训练样本用于下一处循环迁移。

运行约 212 秒，受检权重和个体状态摘要一致，35 项相关测试通过。逐题输出位于 `runs/spora-hybrid-revision-field-order-v1.json`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_recovery runs/spora-composition-late-gates-20-23-24-v1/candidate.pt runs/spora-hybrid-position-recovery-v1 runs/spora-hybrid-revision-field-order-new.json --revision-field-order --recall-transfer data/recurrent-layout-recall-transfer-20261126.json --layout '19,22'
```
