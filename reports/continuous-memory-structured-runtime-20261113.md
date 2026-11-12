# DeepSeek 结构化记忆运行时接入

## 接入内容

`IntegratedDurableRuntime` 现在提供三条受控接口：

- `register_structured_text`：为开放底座个体注册结构化编解码配置。
- `write_structured_text`：把一个身份/事实连续包和边界元数据通过现有 `TensorStore` 事务写入。
- `read_structured_text`：读取不可变副本，避免调用方修改运行时状态。

这些接口与原有神经记忆写入共享修订号、并发检查、崩溃回滚和 bootstrap 校验。结构化记录仍由 `state_codec` 负责序列化，未另建平行持久化系统。

## 真实底座验证

在本地 `DeepSeek-R1-Distill-Qwen-1.5B`、真实 structural bootstrap 和临时 GPU 运行时上完成：

- 写入 4 个实体及其事实。
- 对第一个实体执行一次事实更正。
- 关闭运行时并启动独立进程重载。
- 依据 token id 比较身份和事实，避免中文表面解码的空格/终端编码干扰。
- 四个实体的身份/事实 token 序列跨进程保持一致。
- 更正后的事实 token 序列跨进程保持一致。
- 结构化运行时探针输出：`same_exact_flags: true`，`correction_survived: true`。
- DeepSeek 底座权重和 structural bootstrap 哈希保持不变。

## 失败与修正

第一版探针把更正前的事实继续作为最终期望，错误地把正确更正判为失败。第二版又用中文字符串比较，暴露了 tokenizer 表面解码与 Windows 输出编码不适合作为严格验收依据。最终改为比较 token id，并保留解码文本仅供展示。

## 当前边界

运行时现在能可靠保存和恢复结构化记忆，但写入接口仍接收上层已经解析好的 token 序列和数值地址。自然语言实体识别、事实提取、未知实体拒答和回答生成尚未自动连接到这条写入路径，因此本轮不能称为开放中文聊天能力完成。

## 原始产物

- `runs/continuous-reader-structured-runtime-pilot-v4-20261113/result.json`
- `scripts/deepseek_spora_structured_runtime_pilot.py`
- `spora/integrated_durable.py`
