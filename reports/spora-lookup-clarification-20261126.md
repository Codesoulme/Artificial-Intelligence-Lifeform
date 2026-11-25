# 本机查阅中的追问与改口

学习式整句判定漏掉了多条应查请求，即使使用人工判定，姓名和字段仍会选错。这轮从可明确识别的查阅意图出发，检查让用户补答能否安全完成查阅。PassiveQA 将回答、追问和弃权分别评估。TIDE-Bench 将澄清链和请求改口分开判分，便于验证新的答复是否覆盖了旧目标。 [前一轮对照](spora-read-intent-gate-20261126.md) · [PassiveQA](https://arxiv.org/abs/2604.04565) · [TIDE-Bench](https://aclanthology.org/2026.emnlp-main.2454/)

## 行为

短句含明确查阅动作时，入口保留已提供的姓名、字段和版次，询问下一项缺失信息。后续接受单一的已登记姓名、字段、版次或它们的短组合。如果回复带有多个备选项，会再问用户选择其中一个。明确改口时可替换刚才的选择。若用完整的新地址替换旧地址但没给版次，会重新问版次。补齐才读取张量，重启前未完成的追问只存在于进程内。换话题、取消或清空会结束追问。入口按明确语句规则解析，暂未覆盖隐晦的自然问法。

用两个相似名字开设独立档案，包含饮品、地点和用品的七份新旧记录。测试从无记录开始，经 Windows 正常入口写入。第一进程覆盖缺姓名、缺字段、改口、取消和换地址，第二进程覆盖重启、转题、多个姓名、清空及复查。每次读核对实际记录编号和答复中的名称，每次追问核对所问内容和零读取。默认入口另跑已有的写入与版次回归。

首次回放中的「先别查了」走进普通对话，复述了较早的「果茶」，当轮实际读取列表为空。补全取消语句的识别后，取消会清掉未完成的查阅和本轮对话文本。回放同时检验取消答复不包含档案值。

默认版还出现过两次选对记录却只复述字段名的情况，触发词分别是「见面地点」和「首版」。现在向读出头传递由已确认的姓名、字段和版次组成的完整查阅指令。入口核对生成中复制的名称，再按记录字段显示给用户。完整指令只包含确认过的坐标。

## 结果

| 入口 | 首次启动 | 重启后 | 首次与重启耗时 |
| --- | ---: | ---: | --- |
| 默认版，新增回放 | 24/24 | 13/13 | 35 秒及 38 秒 |
| 循环版，新增回放 | 24/24 | 13/13 | 43 秒及 44 秒 |
| 默认版，原有回放 | 27/27 | 8/8 | 含模型加载 |

新增回放中，每个入口均登记七份张量、完成八次实际读取、发起十五次澄清。补答后的记录编号和答复在两个入口一致。普通的完整地点查阅在默认版答「白鹭堂。」，循环版答「见面地点：白鹭堂。」。改口从沈予安换到沈予岚再查饮品，读回「茉莉茶」。退出前未解答的追问在重启后没有被单独一个姓名触发。两份新档案的摘要保持一致，各自通过 SQLite 完整性检查。59 项版次单元测试通过。

需要澄清的示例采取明确的查档动作和短补答。自然长句、隐含查阅意图和开放中文对话应独立评估。逐条结果位于 `runs/spora-lookup-clarification-default-v2/`、`runs/spora-lookup-clarification-recurrent-v3/` 和 `runs/spora-lookup-clarification-default-control-v1/`。

## 复现

```powershell
$sporaParent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaLayer19 = 'runs/spora-gelu24-recurrent-layer19-v2'
$sporaLayer22 = 'runs/spora-gelu24-recurrent-layer22-v1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_clarification runs/spora-lookup-clarification-recurrent-new --recurrent $sporaParent $sporaLayer19 $sporaLayer22
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_clarification runs/spora-lookup-clarification-default-new
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_version_local_chat runs/spora-lookup-clarification-default-control-new
```
