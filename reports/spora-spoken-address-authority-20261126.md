# 区分局部排除与整句取消

上一轮的五条口语问法都能选出目标人物和事项，动作头没有发起查阅确认。本轮固定原地址模型，在提出地址前比较整句许可与候选专属许可。既有分句实验会把没有提到候选的分句错当证据。此次为不同人标注同一问句下的不同许可，并以完整问句为条件。否定检索研究区分否定对整个请求和对单一候选的影响。Xing USP 分别评估类别和实例绑定。 [上一轮入口实测](spora-spoken-lookup-dialogue-20261126.md) · [既有分句实验](spora-clause-read-20261126.md) · [否定检索](https://aclanthology.org/2025.findings-emnlp.839/) · [Xing USP](https://arxiv.org/abs/2610.34343)

## 对照设计

冻结循环版前 12 层和已有查阅头。保留原人物和事项排序，只更换是否提出查阅的判断。整句组判断整体是否请求任何读取。候选组输入人物位置、事项以及用户整句，判断是否请求读取该候选。同一个正例中，被排除的人仍作为负例。两组使用同样的自编问句和 128 条公开普通请求。候选组将一句拆成两个监督对象，因此训练条数和梯度更新次数更多。校准用不同人名和句式，计分为正确提议减去两倍错地址或多余提议。三组都沿用聊天入口对明确编辑、取消和转题的排除。

新留出包含八对上下文。各对共享姓名和事项，一句只取消其中一人，另一句撤回两人或转为自行处理。另复测上轮的五条口语请求、同句润色和公开普通请求。涉及人物的留出只允许候选真正在测试档案的地址。此阶段比较是否提出正确的确认问题，不读取事实值。

## 动作判断结果

| 指标 | 原查阅头 | 新整句头 | 新候选专属头 |
| --- | ---: | ---: | ---: |
| 校准题中正确提议 | 4/72 | 0/72 | 8/72 |
| 校准题中提议错地址 | 4 | 0 | 0 |
| 新留出中正确提议 | 0/8 | 0/8 | 0/8 |
| 新留出中撤回后仍提议 | 0/8 | 0/8 | 0/8 |
| 上轮口语题中正确提议 | 0/5 | 0/5 | 0/5 |
| 将留出原句改写时错误提议 | 0/8 | 0/8 | 1/8 |
| 拒绝普通公开请求 | 127/128 | 128/128 | 128/128 |

整句头选择了拒绝全部提议的阈值。候选头使用 0.5。它在人工校准取得的收益没有迁移到新问法。在八条新正例中，原头独立挑选人物和事项时，有六条的组合不在档案里。上轮五条口语题的地址均可用，候选头仍未放行其中任何一条。两组各训练 400 和 672 条监督，连同本机加载及评估用时约 53 秒。逐条结果在 `runs/spora-spoken-address-authority-v1/result.json`。

## 按档案目录排序

给定应查许可，固定原模型的姓名和字段分数，比较独立取最高分与先列出已登记组合再排序。目录来自上轮三版档案。每个候选用人物分数与事项分数的乘积排序，这个分数只用于同题比较。补充两种反事实目录，一种为每位人物补齐四个可能的事项，另一种逐题移除正确地址。实验只查看档案元数据。KITE 将可用范围和检索条件分开，读取许可研究同样强调区分匹配和可读取资格。 [KITE](https://arxiv.org/abs/2610.41521) · [读取许可验证](https://arxiv.org/abs/2610.07309)

| 地址候选 | 选对 |
| --- | ---: |
| 姓名及事项各取最高分 | 2/8 |
| 只比较真实档案里存在的组合 | 5/8 |
| 反事实目录中每个人都有四类事项 | 2/8 |

直接接受目录里最高分的记录时，移除正确地址的八条题都会提出替代地址。这一小批里，分数门槛 0.1 或 0.2 留下四条正确地址，其余题及移除目标的题都弃权。门槛扫描仅说明这批数据的取舍。收益来自目录排除了一部分错误组合，档案变密后收益消失。约 32 秒完成，逐条排序在 `runs/spora-spoken-address-authority-v1/catalog.json`。

## 本机入口修正

错提议来自没有引号的「把这句话改得礼貌些：……」。将这类转述和改写识别为文本任务，既有解析器和口语提议都跳过其中的查阅意图。将新规则应用于本批八条改写题，同一组模型分数的误提议从一次降到零。单元测试覆盖「改得礼貌些」「写得更客气」「改成问句」和正常的明示查阅。

真实 Windows 入口通过首次 33 项和重启后 6 项回放。新用例在要求改写「请查罗映禾的见面地点上一版」时进入普通对话，读取列表为空，回应中没有该版的登记值。五次明确查阅依然读取了预期编号，原档案的 12 条记录及摘要一致。62 项版本单元测试全部通过。入口回放在 `runs/spora-spoken-rewrite-boundary-v1/result.json`。

后续要在每位人物具有多项记录的新档案中训练联合排序，监督同一句中正确人物与事项的配对，以及不存在目标时的弃权。普通请求分流和跨轮确认流程可作为对照。此次的动作实验与目录实验继续作为独立对照。

## 复现

动作判断表使用修正文本规则之前的版本。当前代码重复运行时，改写对照使用修正后的判断。

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
$source = 'runs/spora-spoken-temporal-head-v1'
$model = 'runs/spora-spoken-proposal-v1'
$out = 'runs/spora-spoken-address-authority-new'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_spoken_address_authority $model runs/research/xing-usp-883c7dc $out --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_catalog_address_selection $source $model "$out/catalog.json" --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_spoken_lookup_dialogue $source $model runs/spora-spoken-rewrite-boundary-new --recurrent @weights
```
