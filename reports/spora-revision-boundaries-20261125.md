# 完整值起止位置监督

从已有的联合跨度写入器权重起步。同一训练集有 384 条原话，包含 288 条普通提议与确认，以及 96 条新值包含旧值的更正。两组继续训练 5 轮，同样监督选字段和完整跨度。第二组另外提高正确起点、终点在所有合法跨度中的边缘概率。例如，能复制「二十箱」时，起始位置需覆盖「二」，结束位置需保留「箱」。冻结 DeepSeek 孢子、个体状态和跨事项提交决定。 [前轮边界诊断](spora-recorded-value-reranker-20261125.md)

复测前一批 36 条边界题，并在训练前另写 18 条未见原话，来自六组新事实。两份留出无共同目标值，新题的目标值与本轮训练集也无重合。确认和提议分别单测。跨事项仍用原始 34 条原话和 512 条应提交决策，排除只提高单句准确率的情况。

| 留出指标 | 起始写入器 | 联合跨度续训 | 额外监督起止位置 |
| --- | ---: | ---: | ---: |
| 前一批边界题，字段和完整值正确 | 11/36 | 18/36 | 18/36 |
| 新一批边界题，字段和完整值正确 | 5/18 | 4/18 | 4/18 |
| 新一批边界题，截短完整值 | 9/18 | 6/18 | 6/18 |
| 新一批边界题，前 48 个候选包含完整值及正确字段 | 14/18 | 14/18 | 14/18 |
| 原有已确认更正，字段和完整值正确 | 13/24 | 10/24 | 11/24 |
| 原有跨事项原话，字段和完整值正确 | 14/34 | 21/34 | 21/34 |
| 跨事项回放，正确提交 | 8/512 | 12/512 | 12/512 |
| 跨事项回放，错误提交 | 22 | 18 | 18 |

续训在已有边界题上改善了完整提取，但在新一批上少答对一道。额外损失没有给完整值准确率带来相对收益。新题的完整值候选覆盖已经到 14/18，首选仍只有 4 条。扩大可用新值的表达和干扰组合，比在这批合成样本上继续提高起止损失更能检验模型的泛化。

## 与旧值重排组合

独立重载时，把已保存的旧值重排器接到三个跨度写入器上。原写入器加重排，在跨事项 34 条原话中提取对 22 条。联合跨度续训加重排为 27 条，额外监督起止位置加重排为 26 条。给定正确事项后，三种组合的完整提交分别为 336/512、420/512、408/512。实际提交均写对 14/512、写错 16 条。新一批 18 条边界题在三个组合中均只答对 4 条。当前收益受限于已有场景的跨度与跨事项指向，候选继续留在研究分支。

下一步需扩大有真实状态变化的会话样本，并保留独立划分。RiSAWOZ 含中文人际任务对话、逐轮状态、共指和省略标注，可以先核对是否能可靠判断同一事项的更正和完整值区间。本轮未用该语料训练。 [RiSAWOZ](https://aclanthology.org/2020.emnlp-main.67/)

训练与评测约 150 秒，独立重载约 115 秒。重载逐项复现三个主对照和状态回放，并保存旧值重排组合。两次的 32 条受控持久记忆读回均与父实验逐条一致，个体摘要一致。15 项相关单元测试和边缘损失的数值检查通过。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$sporaSplit = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
$sporaArgs = @(
    'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
    'runs/spora-same-field-observation-key-v1',
    'runs/spora-described-revision-span-v1',
    'data/working-state-revision-boundaries-v1.json',
    'data/working-state-revision-boundary-audit-v1.json',
    'runs/spora-revision-boundaries-v1',
    "$sporaSplit/layer-19", "$sporaSplit/layer-22"
)
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_boundaries @sporaArgs
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_boundaries @sporaArgs --verify --prior-reranker runs/spora-recorded-value-reranker-v1
```
