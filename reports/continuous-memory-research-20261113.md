# 持续记忆结构研究与下一轮约束

## 本次核对的研究

- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)：把注意力视为短期记忆，把可训练神经记忆视为长期记忆；长期模块在测试时更新，同时保留并行训练能力。
- [Titan-Llama: Retrofitting Pre-trained LLMs with External Neural Memory Modules](https://arxiv.org/abs/2610.19210)：展示了把层级神经记忆后接到已有语言模型的路线，使用分块注意力保留近期精确上下文，并把更早内容压入固定大小的神经记忆。
- [Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training](https://arxiv.org/abs/2610.36794)：指出统一写入会让无关内容稀释记忆，提出用查询相关性调节每个块的写入强度。
- [Error Correction in Trained TTT Layers](https://arxiv.org/abs/2610.16268)：分析了只做简单 delta 覆盖时预测误差校正不足的问题，说明更新规则必须检查当前预测与目标的残差。
- [DeltaTTT: Layerwise Optimization for Nonlinear Recurrent Memory](https://arxiv.org/abs/2610.08553)：指出非线性记忆的单次串行内循环较难优化，局部目标和分层 delta 更新更容易稳定。
- [Proteus: Incremental Memory Activation for Long-Context Sequence Modeling](https://arxiv.org/abs/2608.16844)：通过随历史增长逐步开放记忆容量，减少早期内容污染后续容量和相互干扰。

## 对 SPORA 的对应关系

已有能力：两处 DeepSeek 层替换、当前/历史神经状态、结构化身份/事实 lane、事务保存和跨进程恢复。

缺口：结构化写入目前要求调用方已给出地址和边界；写入强度没有使用语言模型预测误差；同一有限矩阵长期承载不同内容，缺少渐进扩容和干扰度量；结构化记录还没有接入 Qwen 自回归回答路径。

## 下一轮实验顺序

1. 保持当前正式个体和自然写入入口不变，在隔离候选中增加 query-conditioned write gate。输入包含观察文本和当前预测残差，输出只调节写入强度，不直接决定事实真伪。
2. 用旧事实反事实、无关观察、重复观察和更正观察做 delta 误差对照，要求旧事实保留、重复观察低写入、明确更正可替换。
3. 在相同目标预算下比较固定容量与渐进容量 lane，报告不同实体数量、重复率、旧事实召回和新事实召回。
4. 只有上述结果稳定后，才训练结构化记忆到 Qwen 输出层的条件化解码器；不把结构化 token 重新拼进用户上下文，避免用外部上下文伪造“内部记忆”。

## 当前决策

本轮已经把结构化连续文本通道接入主运行时，但暂不自动开启自然语言解析，也暂不替换正式个体。下一轮优先验证写入学习规则和抗干扰能力，而不是继续堆积中文网页 token。
