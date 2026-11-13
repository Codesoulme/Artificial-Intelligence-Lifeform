# 结构化记忆到文本解码器试验

## 目标

把主运行时的 `3×15` 身份 lane 与 `5×9` 事实 lane 直接输入一个小型自回归记忆解码器，观察它能否在不把记忆 token 拼入用户提示的情况下生成事实句。

## 训练

- 真实 Qwen 词嵌入和主运行时 `90` 维结构化 packet。
- 训练人物：小林、小周、小王。
- 留出人物：小赵。
- 事实：苹果、香蕉、西瓜、葡萄。
- 训练步数：`1000`。

结果：训练集完整句 `9/12`，留出人物完整句 `0/4`；训练集事实内容 `9/12`，留出事实内容 `3/4`。

常见输出把苹果预测为香蕉，并把未见人物小赵输出成训练人物小王。这说明 lane 读取本身没有丢失身份/事实，但小型解码头没有学会从结构化连续表示泛化到完整中文句子。

## 主运行时审计

在隔离 working 个体中写入“小赵 / 苹果”，关闭进程并独立重载，再通过来源哈希校验后的结构化解码头生成。跨进程状态保持一致，但生成结果仍为训练人物和错误事实，审计 `passed: false`。

因此该解码器没有接入默认回答路径。主运行时的结构化记忆读写仍然可用，失败点已经从“保存/恢复”进一步定位到“记忆到自回归文本的泛化”。

## 原始产物

- `runs/deepseek-spora-structured-memory-text-decoder-20261113/result.json`
- `runs/deepseek-spora-structured-memory-answer-audit-v2-20261113/result.json`
- `scripts/deepseek_spora_structured_memory_text_decoder_pilot.py`
- `scripts/deepseek_spora_structured_memory_answer_audit.py`

## 下一步

下一轮应把身份恢复与事实后缀解码分开训练：身份使用 lane 的确定性 token 恢复，解码器只学习事实后缀，并加入未见身份、反事实和未知记录的拒答评测。当前候选不具备默认部署条件。
