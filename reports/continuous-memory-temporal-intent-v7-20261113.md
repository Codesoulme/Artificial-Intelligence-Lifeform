# 读取意图与时间表达迭代

## 变更

- 扩充运行时的当前、历史和后续时间证据词。
- 将读取意图头从 v4 切换到已完成来源校验的 v7 候选。
- v4 保留在 `runs/deepseek-spora-read-intent-gate-v4-20261111/`，作为回退和对照来源。

## 端到端对照

在同一 DeepSeek/SPORA 结构、同一隔离个体、同一 18 条正例和 11 条负例上：

| 读取头 | 正例路由正确 | 正例答案正确 | 负例拒绝 |
|---|---:|---:|---:|
| v4 | 13/18 | 13/18 | 11/11 |
| v7 | 15/18 | 15/18 | 11/11 |

v7 剩余失败为实体跨度或意图置信度不足，未出现新的负例误接受。

## 判定

v7 已替换为默认读取意图头。该结果仍是人工编写、单实体、有限饮料词表的研究评测，不能推导开放领域中文聊天能力。

## 原始产物

- `runs/deepseek-spora-runtime-temporal-generalization-audit-20261113.json`
- `runs/deepseek-spora-runtime-temporal-generalization-audit-v7-20261113.json`
- `scripts/deepseek_spora_runtime_temporal_generalization_audit.py`
- `scripts/deepseek_spora_runtime_temporal_generalization_audit_v7.py`
