# 字段描述能否帮助提取完整更正值

上一轮在 48 条应提交更正中只有 6 条找对了字段和完整值，即使使用正确的提交标签也无法补回其余候选。本轮单独测候选提取，每个案例给出已知场景内的三个字段。比较把字段编号或中文名称放在原话前，观察冻结的孢子特征是否能提供可迁移的字段信息。 [上一轮诊断](20261125-revision-events-stall-diagnosis.md) · [字段描述研究](https://arxiv.org/abs/2201.08904)

两组从同一联合跨度写入器继续，使用 DeepSeek 孢子冻结前 12 层特征、432 条训练原话、相同的采样顺序和 6 轮训练。每条原话按场景的三个字段分别打分。目标字段给出新值的完整字符跨度，其他字段给出空跨度。编号组看到固定数字，描述组看到该场景的中文字段名。训练包含普通更正、提议和确认形式。留出集为原有 24 条更正和对应事实的 24 条提议，原话与训练无重合。自动定位选取跨度相对空操作分差最大的字段。提议中的候选值仅用于提取评测。

| 留出指标 | 字段编号 | 中文字段名 |
| --- | ---: | ---: |
| 已确认更正，给定正确字段时提取完整新值 | 14/24 | 12/24 |
| 已确认更正，自动选对字段 | 18/24 | 18/24 |
| 已确认更正，自动找对字段及完整新值 | 13/24 | 11/24 |
| 提议，给定正确字段时提取完整新值 | 3/24 | 4/24 |
| 提议，自动选对字段 | 17/24 | 16/24 |
| 提议，自动找对字段及完整新值 | 3/24 | 4/24 |

在给定正确字段的提议中，编号组有 8 条原样选中旧值，中文名组为 9 条。两组共同遗漏 20 条完整新值。已确认更正中，给定正确字段后，两组还分别有 6 和 8 条只截取了新值的一部分。中文字段名没有带来一致改善，字段角色和原话中各值的时间角色仍需分别学习。两组共用当前六种场景的候选清单，开放场景字段发现和自动提交仍需独立检验。

在现场提议的下一轮实验中，可以让相同字段、相同两种值分别充当原安排和讨论中的新值，打乱其在原话中的位置。将原记录列为同字段的显式竞争项，同时对不完整跨度做单独监督。采用未见场景与表达划分，比较原始间隔评分和条件式值角色预测。EMNLP 2025 的 From Schema to State 通过先生成对话场景和逻辑流程，再生成话语来扩展结构相同的表达。其结果来自 MultiWOZ，可借鉴数据构造方法，中文迁移需要另测。 [From Schema to State](https://aclanthology.org/2025.emnlp-main.85/)

完整运行约 160 秒，独立重载约 85 秒。重载逐项复现两组候选跨度、评分和持久读取。两次受控持久读取均为 32/32，与父实验逐条一致，个体摘要一致。15 项相关单元测试和字符跨度对齐检查通过。逐题输出位于 `runs/spora-described-revision-span-v1/result.json`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$split = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
$spanArgs = @(
    'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
    'runs/spora-revision-writers-v2',
    'runs/spora-described-revision-span-v1',
    "$split/layer-19", "$split/layer-22"
)
.\.venv\Scripts\python.exe -u -m scripts.train_spora_described_revision_span @spanArgs
.\.venv\Scripts\python.exe -u -m scripts.train_spora_described_revision_span @spanArgs --verify
```
