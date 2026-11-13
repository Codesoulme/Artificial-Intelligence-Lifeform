# 查询相关结构化写入门

## 设计

在结构化身份/事实 lane 写入前增加轻量门控：

- `novelty`：候选 packet 与该地址旧 packet 的余弦差异；重复观察为零。
- `query_relevance`：事实 token embedding 与可选查询文本 embedding 的相关性。
- `write_strength = novelty × query_relevance`。

该门只调节写入强度，不判断事实真假；地址解析、事实提取和来源授权仍由上层负责。默认阈值为 `0.20`，新记录没有旧 packet 时 novelty 为 `1`。

## 真实运行时对照

在隔离 DeepSeek/SPORA working 个体中依次执行：

| 操作 | 写入强度 | 结果 |
|---|---:|---|
| 首次写入“小梅 / 喜欢喝茶” | 1.000 | 写入，revision 1 |
| 重复写入同一事实 | 0.000 | 延期，不产生新 revision |
| 无关观察“喜欢读书”/查询“天气怎么样” | 0.082 | 延期 |
| 更正为“喜欢喝咖啡”/查询“小梅喜欢喝什么” | 0.345 | 写入，revision 2 |

独立进程重载后身份、事实 token、revision 和 checkpoint 哈希均一致，最终事实正确恢复为“喜欢喝咖啡”。审计结果 `passed: true`。

## 判定与边界

这验证了查询相关写入、重复抑制和更正覆盖在结构化连续记忆上的工程闭环。它是 Select-TTT 思路的轻量适配，不是经过训练和校准的事实判断器；embedding 相似度不能证明真实性，也没有接入默认自然语言写入入口。

## 原始产物

- `runs/deepseek-spora-structured-write-gate-audit-v2-20261113/write.json`
- `runs/deepseek-spora-structured-write-gate-audit-v2-20261113/result.json`
- `scripts/deepseek_spora_structured_text_read_after_write_audit.py`
