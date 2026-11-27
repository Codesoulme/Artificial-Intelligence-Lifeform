# 多人对话证据数据核查计划

日期：2026-11-27。前一轮发现 CRECIL 的 `unanswerable` 来自全局关系表缺项，不能作为片段内无证据标签。公开论文 [Molweni](https://aclanthology.org/2020.coling-main.238/) 描述了带可回答及无答案问题、多人对话和篇章结构的阅读理解数据；[MEETINGQA2](https://aclanthology.org/2026.findings-emnlp.1728/) 则强调回答中的说话人归属和事实正确。二者都不证明长期神经记忆能力。

本轮固定 [Molweni 官方提交](https://github.com/HIT-SCIR/Molweni/tree/2eb681d7e610ee3fda4ab3b4cc364554bd17ee67)，仅下载其 `MRC(withDiscourse)/train.json`、README 和许可证至本地 `runs/research/`，校验官方 Git blob 及已公开的 SHA-256；不下载开发或测试集。统计问题类型、答案偏移和对话结构，研究标签如何构造及数据许可。不得将原文或生成样本提交到 GitHub，不执行来源代码。

Molweni 是英语 Ubuntu 技术聊天，其分布与中文人物对话不同。核查通过后可构建训练集内的预先固定分组对照，考察问题与资料共同判断；必须以官方开发集和未用测试集另行验收，并随后检验中文迁移。人工答案位置只用于监督和评价，不提供给实际推理；自然语言生成、持久状态及时间更新仍需另测。
