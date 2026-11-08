# 配对训练顺序工具

`spora.paired_order.PairedOrder` 是独立适配器，不启动训练、不读写文件，也不修改模型。它接收两个可迭代的记录集合，兼容 `scripts/build_matched_dialogue.py` 的 `context.jsonl` 和 `last-user.jsonl` 行：每行包含 `pair_id`、`group`、整数列表 `tokens`、布尔列表 `mask`。调用方负责选择获准使用的训练侧数据、核验来源和划分，并提供记录；本工具不读取或核验清单。

构造时检查每臂标识唯一、集合相同、组相同、有效目标序列一致、首词元不受监督且记录长度不超过 `length + 1`。它按标识对齐，所以输入行顺序可不同。每轮根据种子、轮数和标识的 SHA-256 排序，每对每轮出现一次；跨轮次的顺序是确定的。输入为空、缺对、重复、目标不符或记录过长时立即报错。抽样策略是逐轮无放回，不等同于现有训练器的有放回随机抽样。

```python
from spora.paired_order import PairedOrder, padded

# context_rows 和 last_user_rows 是调用方已载入的训练侧记录。
order = PairedOrder(context_rows, last_user_rows, seed=17, epochs=3, length=511)
batch = order.prepare(batch_size=8)
context_x, context_y = padded(batch['context'], length=511, pad=0)
control_x, control_y = padded(batch['last_user'], length=511, pad=0)
# 将以上两个批次用于同一个完整的双臂更新。
order.commit()
sampler_state = order.state_dict()
```

`padded` 返回普通列表，调用方可转为对应设备的张量；标签与现有对话训练器一样左移一位，忽略位置为 `-100`。应使用相同批量大小、相同划分和相同更新边界，并禁止在生成后替换、截断或随机修改目标；若进行任何变换，训练端须在最终张量上再次比较有效目标以及顺序。不能只比较目标数量。按目标预算累积多个微批时，现有 API 的每个 `prepare` 必须在下次 `prepare` 前提交，因此应在更新开始前保留上一个完整检查点；中途失败必须丢弃内存中两臂及抽样器，整体恢复，而不是保存部分结果。最后不足一个批次时返回剩余全部记录，结束后抛出 `StopIteration`。

每次 `prepare` 返回的 `events` 逐条记录全局位置、从零开始的轮数、标识、组、该标识从一开始的曝光次数、有效目标数量和目标摘要。`commit` 才把事件纳入历史；未提交时重复调用相同批量返回相同内容，不允许导出或恢复状态。返回值是副本，修改它不会改变内部历史。状态包含完整已提交事件，恢复时重新计算顺序与曝光并校验事件、参数、两臂内容摘要；大规模长期运行时，历史和恢复成本随曝光次数增长。

恢复时重新提供相同记录和参数，再调用 `load_state_dict(saved_sampler_state)`。调用方必须把抽样器状态与两臂模型、优化器、学习率调度器、随机状态及更新计数一起存入同一原子检查点，只有双臂完整更新成功后才能保存；检查点中记录最后一次完整更新。进程在任一臂更新后中断时，应整体回到该检查点，不能只恢复抽样器。此模块不管理模型状态、随机算子或分布式执行，也不能单独保证数值结果完全一致。

现有 `scripts/train_conversation_v9.py` 的对话采样及 `spora.paired_experiment.PairedTrainer` 的文档流采样均未接入此工具；真实训练需要改造入口，在同一控制循环中共享批次、对齐累积目标，并按上述方式联合保存和恢复。现有入口及源审核逻辑未被修改。本方案只验证数据顺序和目标，不证明数据可用性或实验效果。
