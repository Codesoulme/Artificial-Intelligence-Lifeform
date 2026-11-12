# 重训自然语言路由闭环

## 变更

将新增的极性头和事实片段头接入隔离压力审计及事务闭环。实体地址头保持原候选不变，DeepSeek 底座、孢子记忆结构和正式个体没有修改。

## 压力审计

在 18 条挑战句上：

- 4/4 正例实体地址通过
- 4/4 正例极性通过
- 4/4 正例事实值通过
- 4/4 正例最终授权
- 14/14 负例拒绝
- 未知实体授权：0
- 未知值授权：0
- 双实体句授权：0
- 跨谓词句授权：0

相较旧头，原先正例最终授权为 `2/4`，新头提升到 `4/4`，负例安全边界保持不变。

## 隔离事务闭环

使用新头执行自然语言到内部记忆的临时工作个体：

- 肯定陈述：`new`
- 否定陈述：`defer`
- 未知事实值：`defer`
- 同一实体事实更正：`correct`
- 独立进程重载后当前值一致：`same_current_value: true`

## 判定

新头已通过比旧头更严格的隔离验证，可以作为下一版研究候选。仍不接入默认聊天入口，原因是测试集规模有限、事实词表有限，尚未覆盖开放世界实体、来源可信度、时间关系和复杂多人关系。

## 原始产物

- `runs/deepseek-spora-natural-route-retrained-stress-audit-20261113.json`
- `runs/deepseek-spora-retrained-natural-observation-transaction-20261113/result.json`
- `runs/deepseek-spora-natural-write-coverage-retrain-20261113/heads.pt`
- `scripts/deepseek_spora_natural_route_retrained_stress_audit.py`
- `scripts/deepseek_spora_retrained_natural_transaction_pilot.py`
