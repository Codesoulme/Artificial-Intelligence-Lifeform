# 显式关系 Lane 试验

上一轮关系迁移失败后，本轮把 packet 的身份槽替换为显式关系槽。实体仍由 memory address 选择，packet 存储关系码和事实值；三个关系域各 4 个训练值、2 个留出值。

入口：`scripts/deepseek_spora_joint_relation_lane_bridge_pilot.py`。

| 指标 | 训练前 | 训练后 |
| --- | ---: | ---: |
| 留出 memory 正确 | 2/42 | 3/42 |
| 留出 swapped 正确 | 0/12 | 0/12 |

按域拆分，训练后 memory 正确数为：饮品 `2`、颜色 `1`、活动 `0`；没有关系域的 swap 侧正确结果。训练仍使用独立关系和值留出，tokenizer 的上下文 span 已通过 offset mapping 处理。

## 判断

显式关系 token 没有解决关系迁移，说明当前问题不只是 packet 缺少关系字段；宿主语言层和关系问法之间仍缺少足够的关系语义学习。该候选不接入，联合水果候选仍是当前最佳受控结果。

本轮未修改正式个体、底座或正式检查点。
