# 运行时自然语言观察接口

## 接口

`SporaChat.observe_text()` 现在提供一个显式 opt-in 的中文观察入口：

1. 使用已加载的自然语言写入头和实体路由解析文本。
2. 生成 `ObservationCandidate` 与校准授权证据。
3. 复用既有 `observe_candidate`、写保护和 working checkpoint 事务。
4. 返回解析结果、动作和持久化结果。

未加载 `NaturalWriteHeadBundle` 时，接口直接拒绝，默认聊天路径保持不变。

## 真实验证

在隔离 DeepSeek/SPORA 工作个体上直接调用运行时接口：

- `小梅偏爱的饮料是茶。` -> `new`
- `小梅不喜欢喝茶。` -> `defer`
- `小梅偏爱的饮料是热巧克力。` -> `defer`
- `小梅偏爱的饮料是咖啡。` -> `correct`

关闭写入进程后重新加载：

- 动作序列保持一致
- 当前事实值保持一致：`same_value: true`
- working checkpoint 哈希保持一致：`same_checkpoint: true`

## 测试

- 运行时聊天持久化：`16/16`
- 自然写入 bundle：`2/2`
- 真实运行时文本观察跨进程闭环通过

## 边界

该接口仍是显式 opt-in。自然语言模型和事实词表的覆盖范围有限，默认聊天入口没有自动开启写入，正式个体没有被本轮实验修改。

## 原始产物

- `scripts/deepseek_spora_runtime_chat.py`
- `scripts/deepseek_spora_runtime_text_observe_pilot.py`
- `spora/natural_write_heads.py`
