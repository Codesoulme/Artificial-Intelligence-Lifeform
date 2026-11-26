# 对照字段描述和直接分类

上一轮中，孢子把多条书目请求认成用品或饮品。本轮沿用第 19、22 层循环注意力候选和已训练的分句选择器。根据 D3ST 的思路，用中文描述四个字段，由同一骨干编码，再让字段头计算所选分句与每个描述的匹配分数。对照组直接输出四类分数。两组使用相同的 160 条去重样本、训练批次和优化设置。训练增加了取自训练集的书目、用品分句，同时纳入四字段成组问法。字段阈值在同一批独立校准题上按同一规则设置。原模型沿用原阈值。 [D3ST](https://arxiv.org/abs/2201.08904) · [SGD-X](https://arxiv.org/abs/2110.06800) · [分句基线](spora-lookup-single-clause-intent-20261126.md)

| 评测 | 上一版 | 重训直接分类 | 描述匹配 |
| --- | ---: | ---: | ---: |
| 原留出正确提议 | 12/24 | 24/24 | 24/24 |
| 原留出错误提议 | 8 | 0 | 0 |
| 第二套措辞正确提议 | 4/16 | 8/16 | 8/16 |
| 第二套措辞错误提议 | 4 | 0 | 0 |
| 第二套措辞零项或双项请求的多余提议 | 0/8 | 0/8 | 0/8 |
| 暂缓与改写请求的多余提议 | 2/8 | 2/8 | 2/8 |

两个新头在通过校准时均选到 0.25 的字段阈值。在只出现一位人物时，四字段首选的概率至少为 0.25，因此这批数据上的拒绝主要由分句意图和候选目录决定。原留出的字段首选从 16/24 升到 24/24，第二套措辞的首选从 8/16 升到 14/16。这项指标包括未通过意图检查的诊断占位分句，实际提议以表格为准。两种新头在这批对照中结果一致。移除正确地址后，两个新头在两套正例上都没有提出别的地址。

第二套措辞的八次漏报都来自同一家族的「说说先前登记的……」。使用这套措辞时，两种字段头都没有收到选中的目标分句。两次暂缓误报来自同一句式「我会另外去确认……」在两位人物上的重复。下一轮应按问法家族分开训练、校准和留出，用成对指令区分请求助手查阅和用户自行确认。四个字段和两位人物同场竞争也需要独立验证，当前留出只覆盖书目与用品的单人查阅。

耗时约 92 秒，输入权重摘要一致。两组权重和逐题数据保存在 `runs/spora-lookup-schema-ranking-v1`，均为离线评测产物。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_schema_ranking runs/spora-switch-address-matched-v1 runs/spora-lookup-field-focus-independent-v1 runs/spora-lookup-schema-ranking-new --recurrent @weights
```
