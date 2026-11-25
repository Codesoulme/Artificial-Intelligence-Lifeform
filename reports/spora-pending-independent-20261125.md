# 在独立对话中检验双重授权

上一轮的两种授权头分别在询问位置和用户转题时误写。按当时提出的对照冻结规则，只有模型判断有待确认提案，且三轮动作头和当前句动作头都同意，才会授权。三个预测头共享原骨干。本轮使用同一份已验证权重，比较单头与同时同意的实际得失。 [上轮写入回放](spora-pending-transactions-20261125.md) · [Hy-MultiTurn](https://arxiv.org/abs/2607.29196) · [否定范围](https://arxiv.org/abs/2610.32281)

## 留出题和写入试验

47 条 CrossWOZ 真实发言中，13 条接受备选，21 条只查尚待决定方案的详情，13 条是在已有方案上追问或进入别的任务。每段来自独立于原授权研究验证样本的对话，继续按原做法遮蔽标注的名称和数字。另编写 12 组接受与不接受的成对发言，覆盖转述和改写引用、否定的目标、尚未满足的条件，并加入三条旧提案对照。提案状态与目标由审核提供。 [CrossWOZ](https://aclanthology.org/2020.tacl-1.19/) · [StateMemBench](https://arxiv.org/abs/2608.19652) · [AFVE-Bench](https://arxiv.org/abs/2611.10541)

九条在开始打分前选定的场景也经过隔离的持久记忆写入。其中三组采用相同的东门提案与初始北门记录，分别改变授权措辞。另外三条使用被替换、已过期和未绑定的编号。模型生成授权决定后，现有提案管理器检查对应会话和版本，成功写入再生成当前和历史答复。该实验使用已知提案地址，自动找出地址另行评估。

## 结果

| 授权方式 | 真实对话中识别接受 | 真实对话中会误授权 | 挑战题中会误授权 | 成对挑战题全对 |
| --- | ---: | ---: | ---: | ---: |
| 三轮动作头 | 11/13 | 1/34 | 8/12 | 2/12 |
| 当前句动作头 | 12/13 | 1/34 | 3/12 | 5/12 |
| 两个动作头同时同意 | 11/13 | 0/34 | 3/12 | 5/12 |

共同误判集中在否定的对象。比如「我确认的是可以查看备选，不是说我要改到八点」，两个头都把对查询权限的确认解读成对改时间的授权。双头在四组否定题的八个句子中只判对两条。引用、条件和否定的分项结果可查 `runs/spora-pending-independent-v1/scores.json`。

预先选定的九条真实写入回放中，两次有效确认写入，两次都能读回新值和旧值。第三次有效确认「我同意东门，不同意继续用原地点」被漏掉。六条不应写入的场景均保留旧值，其中三条被编号检查拦截。三条共同误授权只在离线分类中测到，未纳入预先选定的写入子集。本轮打分约 100 秒，写入和读回约 77 秒，三项提案管理单元测试通过。

## 下一轮实验

自然语言识别先用于生成待确认事项。沿用本地明确确认入口，把确认与对应的提案及当前修订号绑定。Pydantic AI 的 Deferred Tools 将模型请求与调用方按编号给出的决定分开，LangGraph 的 Interrupts 展示了如何携带待确认的内容与结构化回复恢复执行。孢子的现有提案和版本检查已覆盖这一核心流程，可以直接复用。 [本地确认流程](spora-pending-structured-memory-20261124.md) · [Pydantic AI Deferred Tools](https://ai.pydantic.dev/deferred-tools/) · [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)

停止用同一批动作头继续筛阈值。先让原版千问和千问孢子在同一批未参与训练的开放中文对话中逐轮作答，并用人工整理的现行要求作为对照。再在同一个候选上接入明确的记忆写入、重启与历史读回，记录现行要求执行率、被取消内容的复现率、误写次数和重启后的读回结果。这样可以按失分来源决定下一笔训练预算投向普通生成、对话状态还是记忆接口。 [已有底座对照](spora-dialogue-backbone-pivot-20261125.md) · [AFVE-Bench](https://arxiv.org/abs/2611.10541) · [Lost in Conversation 开源评测](https://github.com/microsoft/lost_in_conversation)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$valid = 'runs/crosswoz-awake-recall-val.zip'
$head = 'runs/spora-pending-events-v1/heads.pt'
$out = 'runs/spora-pending-independent-v1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_pending_independent $valid $head $out
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_pending_independent $valid $head $out --apply
```
