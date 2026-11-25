# 人工中文对话中的时间误确认与自动记忆寻址

前一轮按意图和候选分别判断，能在自编题上拦住误确认，同时漏掉许多已确定时间。本轮用独立的公开中文对话检验误确认，并把普通请求的自动寻址单独推进。时间实验冻结三个候选。寻址实验训练单独的四字段头，沿用 DeepSeek 和版本快权重。 [时间对照](spora-time-authority-20261125.md) · [StateMemBench](https://arxiv.org/abs/2608.19652) · [TransMem](https://arxiv.org/abs/2611.10683)

## RiSAWOZ 查询是否被当作安排

从固定版官方开发分区选 26 个有明确时间候选的用户回合，逐条核对用户当前话语和邻近上下文。包括查天气、查班次、按时间改查和比较备选。举例来说，用户改变想查询的动车时间，仍处于查找交通选项的阶段。某些查询带有已知个人计划，但当前任务是查信息。此处的目标是避免把查询日期当作已授权写入通知的日期。三种模型都只看同样的当轮原话。本组全部是时间安排的负例，只能测误确认。查询所需的临时时间另有用途。公开档案及原话保留在本地，仓库中的评测脚本只附上开发分区和片段索引。 [RiSAWOZ 固定发布版](https://github.com/terryqj0107/RiSAWOZ/tree/1e2d3be2b25a0540519db1664a172d85ddd62b29) · [数据论文](https://aclanthology.org/2020.emnlp-main.67/)

| 时间模型 | 将查询或备选误当成正式安排 |
| --- | ---: |
| 原选择器 | 18/26 |
| 同结构的配对训练 | 9/26 |
| 分别预测意图与候选 | 8/26 |

双头仍误认了一条追加查询下周四动车、一条带两种上课日期偏好的找班请求，也误认了普通电影推荐中提到的「周末」。这一结果与自编负例的 0/14 明显不同。后续时间训练应提供日期查询、确定安排及条件未满足的对照语料，并保证先前明确安排的正确接受率。时间对照用时约 50 秒，CUDA 张量峰值约 3.37 GiB。

## 普通请求能否自动读到内部字段

四字段头在冻结 DeepSeek 前 12 层的句末表示上训练 12 轮，共用 156 条自编请求。标注字段为见面地点、用品、饮品、书目。测试使用六位不在训练中的人物、24 份原有版本张量和 23 条不同措辞的问句。13 条要求读取一项或两项，10 条是文字处理、泛聊、换方案、未登记人物或多人物。实验范围为单一已登记人物的当前版。根据可见姓名提出候选人物，根据训练头预测字段。只在候选人物唯一、所有选中字段都存在及数量符合实验范围时读回。普通对话没有提供任何已存值。已有显式寻址用空历史和能直接解析的当前版作对照。 [已有版本入口](../docs/version-chat-local-use.md)

| 结果 | 已有显式寻址 | 训练后的字段头 |
| --- | ---: | ---: |
| 完整选出应读字段 | 1/13 | 4/13 |
| 在本次无需自动读取时误读 | 0/10 | 0/10 |
| 选对字段且实际生成全部正确值 | 未运行 | 4/13 |

四次成功的读回包括读取书名、碰头地，以及两个饮品加书名的组合。另有一次把饮品请求误判为书目，一次在需要书目和地点时只读了地点。其余七条该读未读。一次明确写了用品当前版的指令也被新头漏掉，说明学习模型只能继续作为自然表达的候选，显式寻址应先保留已有解析。此次读回使用原短答头，所以成品句能力没有计入该项。版本库摘要前后相同，49 项版次测试通过。训练与评测约 99 秒，CUDA 张量峰值约 3.47 GiB。独立重载约 94 秒，复现 23 条判定和其中六次实际读出。逐题结果见 `runs/spora-composition-addresses-v1/` 和 `runs/spora-risawoz-time-intent-v1/`。

后续将字段证据与整句意图区分。例如「那一款」能否和同句的「常喝」绑定到饮品，不能由整句里缺少「饮品」二字决定。下一轮按字段候选和对应短语来评分，同时分别测试明确读取、泛聊、无记录、不同人物的同名字段。通过后再接入成品生成和重启回忆。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaScope = 'runs/spora-composition-addresses-new'
$sporaReader = 'runs/spora-dual-version-reply-v1'
$sporaRecords = 'runs/spora-composition-training-v1'
$sporaCorpus = 'runs/research/risawoz-1e2d3be'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_composition_addresses $sporaReader $sporaRecords $sporaScope
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_composition_addresses $sporaReader $sporaRecords $sporaScope --verify
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_risawoz_time_intent $sporaCorpus runs/spora-active-time-v2 runs/spora-time-authority-v2 runs/spora-risawoz-time-intent-new/result.json
```
