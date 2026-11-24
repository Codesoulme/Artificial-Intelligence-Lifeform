# 用已有记录辅助提取更正值

从同一冻结写入器取每个预测字段分数最高的 48 个合法跨度。对照保持原始排序、在有其他候选时跳过等于旧值的跨度，以及用小型重排头学习旧值和跨度的关系。重排头使用原跨度相对分数、位置、标点边界和已存值的字面重合程度。它不改写原话，事项和提交决定沿用上轮登记向量实验中的逐题分数。测试提供候选草稿所属的事项。 [跨事项基线](spora-same-field-reference-20261125.md) · [TripPy](https://aclanthology.org/2020.sigdial-1.4/)

用 432 条训练句和 9 轮拟合重排头，训练句包括同一个值分别充当旧值和备选值的成对样本。训练时可补入正确跨度以提供监督。评测只从冻结写入器选出的候选中重排。跨事项的 34 条原话中，冻结写入器在 33 条找到正确字段并将完整新值纳入候选集合。

| 留出指标 | 原排序 | 跳过等同旧值 | 学习重排 |
| --- | ---: | ---: | ---: |
| 跨事项原话，完整值和字段正确 | 14/34 | 16/34 | 22/34 |
| 跨事项原话，原样复制旧值 | 9/34 | 0/34 | 1/34 |
| 跨事项原话，截短完整新值 | 4/34 | 4/34 | 8/34 |
| 独立提议，完整值和字段正确 | 3/24 | 5/24 | 9/24 |
| 独立已确认更正，完整值和字段正确 | 13/24 | 13/24 | 11/24 |
| 跨事项状态回放，正确提交 | 8/512 | 8/512 | 14/512 |
| 跨事项状态回放，错误提交 | 22 | 22 | 16 |
| 给定正确事项后的跨事项回放，正确提交 | 208/512 | 236/512 | 336/512 |

两种改动都减少了原样复制旧值。学习重排能从相同候选中找回更多完整值，但截短案例增加。只在原话含有已存旧值时采用重排的补充对照，保留了跨事项和提议题的提取收益，已确认更正恢复为 13/24。这个启用条件是看到这一批确认题退步后提出的，仍需在全新的表述和同值子串上检验。固定事项的结果用于分离提取与指向能力，自动回放仅完整写对 14/512。

这轮主要改善的是已有记录辅助复制的通路。下一步需要继续训练完整值边界，再与跨事项匹配一起测自然对话中的整条记录。当前候选只在研究分支验证。

重排头有 449 个参数，权重文件约 3.6 KiB。训练和评测约 104 秒，独立重载约 90 秒。重载复现三个主对照和状态回放，并单独生成按旧值证据启用的结果。两次的 32 条受控持久记忆读回均与父实验逐条一致，个体摘要一致。15 项相关单元测试及旧值子串和回退条件检查通过。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$sporaSplit = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
$sporaArgs = @(
    'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
    'runs/spora-same-field-observation-key-v1',
    'runs/spora-described-revision-span-v1',
    'runs/spora-recorded-value-reranker-v1',
    "$sporaSplit/layer-19", "$sporaSplit/layer-22"
)
.\.venv\Scripts\python.exe -u -m scripts.train_spora_recorded_value_reranker @sporaArgs
.\.venv\Scripts\python.exe -u -m scripts.train_spora_recorded_value_reranker @sporaArgs --verify
```

主对照写入 `result.json`，根据旧值证据启用重排的结果写入 `old-evidence-gate.json`。
