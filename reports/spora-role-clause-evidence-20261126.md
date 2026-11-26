# 按字段定位查档分句

对已有留出逐题检查后，36 条应查请求里有 11 条完全未读。另有 15 个应读字段在已触发读取的请求中被漏掉。已读到的 29 个目标字段中有 10 个选错版次。新留出的 24 条应查请求中，相应数字为 7 条、11 个及 8/16。三个位置都存在失误。 [分开判断查档、角色与版次](spora-role-version-routing-20261126.md)

D3ST 用自然语言描述匹配字段，STITCH 按当前意图限定可用记忆。本轮沿用冻结 DeepSeek 的字段描述表示，按标点切分请求，让每个字段从分句中选择证据。整句表示负责判断查档意图，整句与分句表征共同决定字段和版次。训练分句标签来自自编训练句中的字段别名，留出只输入请求文字与候选字段名称。版次库及低秩张量只在选择后进入成句流程。 [D3ST](https://arxiv.org/abs/2201.08904) · [STITCH](https://aclanthology.org/2026.findings-acl.584/)

两轮分句实验与整句基线共用 216 条训练、42 条校准、48 条旧留出、36 条新留出、220 步优化和相同冻结输入。第一轮构建分句头前消耗了随机数，导致共享层初始值不同。第二轮修正实例化顺序，在与基线相同的共享层初始值上复核。下表使用当时相同的查档资格过滤，格内依次为旧留出、新留出。

| 指标 | 整句基线 | 分句，首轮 | 分句，相同共享层初值 |
| --- | ---: | ---: | ---: |
| 应查题中完全未读，越低越好 | 11/36 · 7/24 | 12/36 · 8/24 | 14/36 · 9/24 |
| 应查题中字段集合全部正确 | 9/36 · 6/24 | 11/36 · 8/24 | 7/36 · 7/24 |
| 应查题中字段和版次全部正确 | 3/36 · 2/24 | 8/36 · 3/24 | 5/36 · 4/24 |
| 已选中的目标字段选错版次 | 10/29 · 8/16 | 9/28 · 7/17 | 6/23 · 5/16 |
| 答复事实完整、无竞争值且正常结束 | 6/36 · 4/24 | 8/36 · 8/24 | 6/36 · 7/24 |
| 无需查档题误读，越低越好 | 0/12 · 0/12 | 1/12 · 0/12 | 1/12 · 0/12 |

相同初值时，新留出多答对三题，旧留出持平。读到的目标字段选错版次减少，但该分母随路由改变。在同一组目标字段上，分句头门控前的版次判断分别答对 45/60、24/36，基线产物未保存门控前版次。分句在两轮均增加了整题漏查。受控具名读回在每个留出均为 32/32。两轮分别约 188、179 秒，源权重、实验版本库摘要和个体摘要与基线一致。新旧留出沿用相同六类事项与已确认对象，生成判分核对词面事实和结束符。

两轮误读都来自同一条以「不需要翻档」开头的请求。读取边界现在识别这种独立的取消句。试聊入口清除待确认项和近期查阅文本后继续处理本轮普通请求，带明确字段的局部排除仍可查其他项。用保存的逐题动作复查，新边界各拦下一次读取，正例动作没有变化。单元测试覆盖取消和局部排除、上一轮地址与值的清理、字段换序及分句填充遮罩。

Windows 默认入口回放通过首次启动的 25/25 项和重启后的 13/13 项，分别耗时约 45、53 秒。新增例在实际读过首版饮品后要求不翻档，选中的记录为空，普通回答未复述七条保存值。重启前后实验档案完整性、记录数和摘要一致。112 项相关单元测试通过。首次回放未收到入口回复，重新启动后完成全套校验，结果在 `runs/spora-lookup-clarification-archive-cancel-v2/result.json`。

分句候选仍在研究入口。TIDE-Bench 将补齐缺项及更正意图分别评分，LongMemEval 独立测量时序、更新与弃权。下一轮用多种用户改写建立跨事项留出，将「明确无需查档」「明确要查但缺少字段或版次」「字段及版次都明确」分别测量。对中间组使用现有的补问和确认入口，对最后一组继续对比可解释的字段证据。 [TIDE-Bench](https://aclanthology.org/2026.emnlp-main.2454/) · [LongMemEval](https://arxiv.org/abs/2410.10813) · [本机补问评测](spora-lookup-clarification-20261126.md)

## 复现

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$parent='runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$reference='runs/spora-revision-version-state-v1/result.json'
$reader='runs/spora-revision-roles-shared-v2'
$layout='runs/spora-hybrid-position-recovery-v1/19-22'
$layers=@("$layout/layer-19", "$layout/layer-22")
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_role_version_routing $parent $reference $reader runs/spora-role-version-routing-clause-replay-v1 @layers --clause-evidence
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_clarification runs/spora-lookup-clarification-archive-cancel-replay-v1
```

当前命令使用更新后的读取边界。两轮原始分句评估保存在 `runs/spora-role-version-routing-clause-v1` 和 `runs/spora-role-version-routing-clause-v2`。
