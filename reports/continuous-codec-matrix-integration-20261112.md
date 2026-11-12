# 连续词元编码与孢子矩阵集成（2026-11-12）

将 `ContinuousTextCodec` 的连续词元代码写入临时孢子快速矩阵，再从两处矩阵读取并通过同一个码本解码。使用 16 条随机中文词元序列、数值正交地址和固定码本；首次写入后 16/16 条在两处矩阵都完整恢复。覆盖第一个地址后，新值在两处恢复正确，其余 15/15 条在两处仍完整恢复。

集成中发现并修复两个接口缺口：CPU 随机生成器不能直接用于 CUDA 张量，现先在 CPU 生成投影再转移；底座词嵌入宽度为 1536，而孢子矩阵宽度为 90，现允许显式传入 `memory_width` 并按它确定槽宽。新增单元测试覆盖较窄记忆宽度，四项组件测试全部通过。

本次仍是隔离实验：地址由程序指定，写入绕过正式来源控制、历史、持久化和自然语言理解；解码依赖底座词嵌入生成的共享码本，不是语言模型自己自由回答。只覆盖 16 条短序列和一次覆盖，不能证明开放领域、未知项拒答或长期记忆可靠。默认聊天和真实个体没有接入此组件。

代码 `spora/continuous_text_memory.py`，测试 `tests/test_continuous_text_memory.py`，集成脚本 `scripts/deepseek_spora_continuous_text_codec_integration.py`；本机结果 `runs/deepseek-spora-continuous-text-codec-integration-retention-20261112.json`。源底座和检查点经摘要核验未变。
