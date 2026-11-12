# 自然语言写入头正式化

## 改动

新增 `NaturalWriteHeadBundle` 和来源校验加载器，将重训后的极性头、事实片段头及阈值封装为版本化研究组件。加载时校验：

- bundle 格式和张量形状
- DeepSeek 底座权重哈希
- 固定结构来源 checkpoint 哈希
- SPORA 替换层 feature variant 哈希

运行时支持通过 `natural_write_heads_path` 显式加载 bundle。默认路径不传该参数时保持原有头不变。bundle 在 CPU 上运行，与原特征提取路径的设备契约一致。

## 验证

- bundle 单元测试：`2/2`
- 运行时聊天回归：`14/14`
- 时间路由回归：`2/2`
- 使用显式 bundle 的自然语言压力集：正例授权 `4/4`，负例误授权 `0/14`，未知实体/未知值/双实体/跨谓词均为 `0`。
- 使用显式 bundle 的自然语言事务闭环：`new / defer / defer / correct`。
- 独立进程重载后：`same_current_value: true`。

## 关键修复

首次独立重载把演化中的 working checkpoint 用作 head 来源校验，导致 provenance mismatch。已改为始终使用不可变结构来源 checkpoint 校验，working 文件只承担个体演化状态。这一分离是持久化运行时的必要边界。

## 当前边界

bundle 仍是研究组件，未设为默认聊天入口。路由数据规模、事实词表和开放世界实体覆盖仍有限，后续需要扩大独立中文评测后再决定默认启用。

## 相关文件

- `spora/natural_write_heads.py`
- `scripts/deepseek_spora_runtime_chat.py`
- `scripts/deepseek_spora_natural_observation_transaction_pilot.py`
- `scripts/deepseek_spora_natural_route_retrained_stress_audit.py`
- `scripts/deepseek_spora_retrained_natural_transaction_pilot.py`
