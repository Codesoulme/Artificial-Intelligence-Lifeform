# TTT 记忆路线研究与落地判断

## 查到的研究

- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) 将短期注意力与测试时神经记忆结合，强调记忆模块需要与语言网络共同训练。
- [MEMORYLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624) 和 [M+: Extending MemoryLLM with Scalable Long-Term Memory](https://arxiv.org/abs/2502.00592) 将可更新状态放入模型层内，并通过训练让生成路径认识这类状态；M+还加入长期检索器。
- [Test-Time Training with Next-Token Prediction](https://arxiv.org/abs/2606.21803)（EMNLP 2026 Findings）把每次写入的 value 目标设为下一位置的上下文隐藏状态，并把 fast weight 放在原始 MLP 的 down projection 中。其公开代码给出的关键配置包括 `ttt_target: hidden_states`、`ttt_predict_mode: next` 和行范数保持。
- [MemDefrag](https://arxiv.org/abs/2607.05969) 专门处理多次 latent memory 更新后的碎片、位置编码失配和相关片段追踪，说明长期记忆不能只做矩阵累加，还需要检索、重排和遗忘。
- [Titans-as-a-Layer](https://arxiv.org/abs/2606.08573) 将测试时记忆作为与输入 token 对齐的层适配器读回宿主模型，支持“记忆作为 residual”这一接入形式。
- [MoNe: Modular Neural Memory](https://arxiv.org/abs/2608.17616) 采用分段写入和查询时读取，强调预处理成本与查询成本分离。

## 对当前 SPORA 的影响

当前有限值头的路径是“记忆索引 -> 固定饮品词元”。它适合验证读取链路，但不能保存任意中文事实。此前精确词向量桥和注意力分支也没有解决开放泛化，原因是它们的连续表示没有经过与宿主语言层一致的写入训练。

本轮采用 TTT-NTP 的核心思想，但按本机显存做了缩小版：保留 DeepSeek 原始 28 层和 MLP，选择第 6、12 层的 `down_proj`，用事实文本的 down-projection 激活作 key，用下一位置的同层上下文隐藏状态作 target，通过 rank-one fast-weight 更新形成临时个体记忆。这个状态不写入底座检查点。

初始零训练 graft 的 3 个写入强度均不能召回正确事实，最高只改变了一个错误答案。这说明论文方法不能直接零样本移植到未经过 TTT 目标训练的 DeepSeek 检查点。

随后增加两个低秩语言读取适配器，冻结 DeepSeek 底座和 fast weight，只训练“TTT 读向量 -> 宿主隐藏残差”的映射。96 步后，8 个训练人物/事实从 0/8 提升到 8/8；4 个未见人物从 0/4 提升到 2/4；无记忆普通请求保持不变。这个结果首次证明当前项目的记忆内容可以通过上下文状态写入，再由小型语言适配器接回原生生成路径，而不依赖有限值词表。

## 当前结论

这条路线值得继续，是研究分支的下一候选，但还不能接入正式个体。未见人物的 2/4 说明读取表示仍含有身份和表面句式偏差；多条关系、历史纠正、自然写入和跨进程恢复尚未验证。下一轮应扩展事实表面形式和问题问法，加入多实体同一状态、当前/历史切换、未知事实拒答，并比较 TTT 写入的检索相似度和适配器输出。

代码和结果：

- `spora/research_ttt_ntp.py`
- `scripts/deepseek_spora_ttt_ntp_pilot.py`
- `scripts/deepseek_spora_ttt_read_adapter_pilot.py`
- `runs/deepseek-spora-ttt-ntp-pilot-v2-20261114/result.json`
- `runs/deepseek-spora-ttt-read-adapter-pilot-20261114/result.json`

所有实验均使用临时 fast weight 和临时适配器；DeepSeek 原始权重、默认检查点和正式个体没有修改。
