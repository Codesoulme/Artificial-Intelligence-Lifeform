# 按分句读取多轮更正

固定 DeepSeek 孢子前 20 层，在标点与换行处取分句末尾的隐藏表示。平均汇聚组使用分句均值和最后一个分句的表示。循环组使用相同均值，并让一个 64 维 GRU 按顺序处理分句，再取最后的循环状态。循环组多 24,960 个参数。训练量、学习率和随机种子相同。两组各从随机初始化训练 10 个轮次，使用 1,152 条单句和 576 对末句相同的多轮训练请求。门槛取独立校准集的最高负例分数。[整句对照](spora-write-intent-20261124.md)

| 条件 | 多轮写入正确 | 多轮负例误放行 | 最新决定写入正确 | 最新决定负例误放行 | 最新决定正确排序 |
|---|---:|---:|---:|---:|---:|
| 分句平均汇聚 | 1/12 | 0/12 | 0/8 | 0/8 | 7/8 |
| 分句循环处理 | 4/12 | 0/12 | 1/8 | 0/8 | 5/8 |

循环候选在保守门槛下多通过四条合法请求，最新决定的排序少对两对。它在校准集的 96 条合法请求中也只放行了 3 条。结果来自合成短对话与固定中间回复，当前权重作为离线候选保存。训练与评测使用不同姓名和写法。

近期研究提供了两种直接可复用的验证方法。AgentMemGate 在候选事实确认之前检查记忆是否已被污染，并区分确认与放弃。StateMemBench 用可执行的提交、更新和撤回序列计算每轮的应有状态，也检查后续提及没有取代旧值的情况。孢子可沿用 P24 的工作状态承载未确认候选，再用独立的确认或撤回操作改变状态。验证应在提议后、确认后和撤回后实际读取槽位，同时检查未涉及字段的内容。[AgentMemGate](https://arxiv.org/html/2610.07707) · [StateMemBench](https://arxiv.org/html/2608.19652) · [P24 工作状态](../docs/P24-awake-design.md)

独立进程重载逐题复现结果，源模型和个体状态摘要一致。权重与记录位于 `runs/spora-clause-intent-v1/`。

```powershell
$sporaModel = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
.\.venv\Scripts\python.exe -m scripts.audit_spora_clause_intent $sporaModel runs/spora-clause-intent-v2
.\.venv\Scripts\python.exe -m scripts.audit_spora_clause_intent $sporaModel runs/spora-clause-intent-v2-reload --verify runs/spora-clause-intent-v2
```
