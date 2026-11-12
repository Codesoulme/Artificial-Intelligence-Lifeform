# CLI 中文观察入口

## 入口行为

本地运行时 CLI 新增：

- `--observe-text`：提交一条中文观察。
- `--natural-write-heads`：显式指定带来源校验的自然语言写入头 bundle。

两者必须同时出现。没有 bundle 时 CLI 在加载模型前拒绝，默认问答和原有数值观察入口不变。

## 验证

在隔离 working 个体上直接执行 CLI：

`小梅偏爱的饮料是茶。`

结果：

- 实体地址：`16/16`
- 极性：`affirmative`
- 事实值：`茶`
- 动作：`correct`
- 持久化结果：`current_write: true`

API 级运行时测试 `17/17`，自然写入 bundle 测试 `2/2`。无 bundle 的 CLI 安全门也已验证。

## 边界

CLI 写入入口仍是显式 opt-in，只用于研究 bundle 和隔离个体。正式默认个体不会被自动写入，开放世界事实和更广泛中文表达仍需继续评测。
