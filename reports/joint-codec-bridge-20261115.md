# 联合 Codec 与隐藏桥接训练

> **勘误（2026-11-15）：** 后续审计发现，当时保留梯度的快照复制仍处于 `torch.no_grad()` 中；五个后续候选的 codec projection 完全相同，故本报告中“codec projection 参与语言损失”和“更新后的 codec”是不成立的。记录的生成分数和重载一致性仍是原实验的实测结果，但不能归因于 codec 参数联合更新。修复与后续诊断见 `joint-codec-gradient-repair-20261115.md`。

本轮首次让连续记忆 codec projection 参与语言损失：

1. codec 在写入时使用可微 token projection；
2. 时间写入在研究模式下保留 target 梯度；
3. 集成 memory snapshot 在研究模式下不 detach；
4. 读取器、记忆桥和语言适配器共同优化。

相关入口：

- `scripts/deepseek_spora_joint_codec_bridge_pilot.py`
- `scripts/deepseek_spora_joint_codec_bridge_warmstart_pilot.py`
- `scripts/deepseek_spora_joint_codec_bridge_continuation_pilot.py`
- `scripts/deepseek_spora_joint_codec_bridge_reload_audit.py`

## 结果

第一次联合试验因重新初始化语言桥，训练前仅有 `75/108` 严格配对，不能用于比较。随后从最佳线性桥 warm-start，并恢复 codec state：

| 指标 | Warm-start 前 | Warm-start 后 | 320 步延续后 |
| --- | ---: | ---: | ---: |
| 留出人物记忆 | 111/120 | 115/120 | 115/120 |
| 留出人物换值 | 99/108 | 103/108 | 103/108 |
| 严格反事实配对 | 91/108 | 92/108 | 99/108 |

最终候选：`runs/deepseek-spora-joint-codec-bridge-continuation-20261115/`。

独立进程重载审计：

- 已记录值逐词 parity：18/18
- 新人物：10/12
- 新问法：4/8
- 多事实共存：6/6
- 更正后读取：6/6
- 关闭记忆控制：0/8 条被误判为事实正确

延续训练约 826 秒，峰值 CUDA 分配约 4.66 GiB。候选保存了更新后的 `codec_state`，重载审计已实际加载并验证。底座、正式个体和正式检查点未修改。

## 判断

这是目前第一个有独立重载证据的联合 codec/语言桥改进：严格反事实配对从线性桥的 `93/108` 提升到 `99/108`。它仍使用受控地址、固定水果关系和有限问法，新问法泛化只有 `4/8`，因此不能称为正常聊天突破，也不接入正式个体。

下一步应把同一联合目标扩展到关系和值的独立留出集，并保持 codec state、崩溃恢复和普通无记忆路径的独立审计。
