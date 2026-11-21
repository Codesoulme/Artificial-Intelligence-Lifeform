# 中文能力归因与底座决策

日期：2026-11-21

## 结论

当前中文能力缺失由两个因素共同造成：

1. 整层孢子替换破坏了原语言骨干的函数；
2. `DeepSeek-R1-Distill-Qwen-1.5B` 本身不是成熟中文聊天底座，具有明显的推理格式、固定拒答和不稳定多轮承接先验。

两者中，整层替换是确定的结构性回退来源，DeepSeek 底座是当前中文聊天上限偏低的主要来源。

## 证据

原始骨干与整层替换的同进程对照：

| 条件 | 原始 DeepSeek | 整层孢子替换 |
|---|---:|---:|
| C-Eval | 24/60 | 18/60 |
| 严格双轮对话 | 1/8 | 0/8 |
| 受控记忆 | 7/8 | 7/8 |

记忆读取保持不变而语言指标下降，说明回退主要发生在语言层函数保持，而不是记忆矩阵本身。

同一份 `conversation-development-v2` 留出集的底座对照显示，Qwen2.5-0.5B-Instruct 能在普通聊天模板中处理部分先后顺序、数量更新、简短改写、未知事实和否定变化；DeepSeek 原始骨干则频繁输出固定拒答、联网建议和无关说明。Qwen 仍会混淆人物关系、改约信息和条件计划，因此它不是最终模型，只是更适合验证聊天和记忆接入的语言控制组。

## 残差实验

保留 DeepSeek 原始语言骨干、只在第 9、15、21、27 层增加零初始化秩 8 残差后：

| 指标 | 零初始化 | 关闭思考残差 |
|---|---:|---:|
| C-Eval | 24/60 | 23/60 |
| 对话正确轮次 | 3/16 | 8/16 |
| 自然结束轮次 | 13/16 | 16/16 |
| 记忆正确 | 7/8 | 7/8 |
| 记忆切换 | 6/8 | 5/8 |

独立留出中，候选改善了日程修改、条件计划、数量更新和部分事实承接，但仍会违反“不加鸡蛋”、混淆人物关系，并未恢复自然思考模式。

加入 C-Eval 原始分布保护项的变体反而得到 `22/60`、对话 `6/16`、记忆切换 `5/8`，不晋级。把残差强度降到 `0.5` 或 `0.75` 也没有恢复 C-Eval，分别为 `22/60`；完整强度才保留 `8/16` 对话收益。

## 决策

- 不再继续整层替换语言骨干。
- DeepSeek 分支保留，用于研究思考段和推理行为。
- Qwen2.5-0.5B-Instruct 作为中文聊天和孢子记忆接入的主验证底座。
- 孢子记忆、时间路由和状态通道继续以零初始化旁路接入，不能覆盖原始语言主干。
- 当前 DeepSeek 残差候选和 C-Eval 保护候选均不进入正式检查点。

相关结果：

- `runs/diagnostic-original-all-control-20261121-v3.json`
- `runs/awake-closed-reviewed-dialogue-residual-v1/result.json`
- `runs/awake-closed-reviewed-dialogue-residual-v1/holdout.json`
- `runs/awake-closed-reviewed-dialogue-ceval-guard-v1/result.json`
- `runs/awake-closed-reviewed-dialogue-residual-v1-scale-sweep.json`
- `runs/qwen2.5-0.5b-instruct-holdout-20261121.json`
