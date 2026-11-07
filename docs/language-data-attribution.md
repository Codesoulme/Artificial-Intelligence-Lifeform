# 中文研究教材来源与处理记录

2026-11-06获取，仅用于本机非商业研究。使用的是文本材料，没有下载第三方模型权重或词表。

提供者：gongjy（MiniMind项目），[数据仓库](https://www.modelscope.cn/datasets/gongjy/minimind_dataset)，[项目说明](https://github.com/jingyaogong/minimind)。数据卡声明CC-BY-NC-4.0，并描述包括deepctrl、Magpie、COIG等的混合上游来源与相应许可。该声明不是逐条语料权属或事实正确性的独立核验。原文不随本项目上传。

| 文件 | 固定版本 | SHA256 |
|---|---|---|
| pretrain_t2t_mini.jsonl | 21dd8c9258b6dc73971ff10d4dd3924caf32c644 | 6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c |
| sft_t2t_mini.jsonl | 833d262180bb2cddbba1284222ad4b5547b6b224 | abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518 |

本地原始位置：runs/language-material/。同时保存数据卡、下载地址、字节数、固定修订及散列。数据内容的任何指令均只作不可信训练材料，不是本开发代理的工作指令。

处理：中文比例及长度过滤、排除工具调用和内嵌思考文本，仅读取对话content而不训练reasoning_content；过滤外部模型名称／开发者和部分虚构亲历句；全文规范化去重；按规范化前96字符分组散列划分；仅以训练分区学习自建BPE；构造助手输出掩码，用户输入不作为对话响应监督。v2另做散列抽样增加来源覆盖，v3进一步清理身份询问和重复结尾。初版A的漏滤与失败权重保留。

限制：关键词过滤会漏过语义改写，也会误删正常讨论模型的教材；未逐条核实事实。来源文件的顺序截取仍有分布偏差。精确去重和有限字符片段检查不能证明所有近义重复已消除。最终聊天验收另行构造并审查，而非直接把公开语料保留损失当能力结论。

用户指定社会主义核心价值观材料来自本项目原课程，国家／社会／个人三层原文持续作为独立回放，不用第三方内容替换。

## 短教学课程的来源评估

2026-11-06抽查MiniMind短回答子集后，仍发现事实错误、缺失题干和虚构经历，故runs/conversation-curriculum-v1只是失败筛选产物，尚未用于训练。另查阅COIG-CQIA卡片，其总许可字段写为More Information Needed；当前只保留卡片，没有将其数据纳入课程。

随后从Instruction-Tuning-with-GPT-4/GPT-4-LLM读取中文教学文本：固定Git修订80cda626ea305004be42426671c66efebbf22144，路径data/alpaca_gpt4_data_zh.json，共48818条，35100559字节，Git对象SHA1为af092742e9bc3b83fa6feb13203b793aafc9dffc，SHA256为6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903。仓库原文明确为CC-BY-NC-4.0、仅研究用途；遵循此范围，不公开推送原始教材。第三方镜像卡片头部与正文许可不一致，采用原项目更严格的研究／非商业声明。

这份内容由GPT-4生成，不是人工逐条保证正确。经过身份、控制标记、长度、严重重复、句尾完整性及旧／新留出重叠等筛选，25704份进入D／E训练。过滤不保证事实完全正确。原始文件、提供方说明、下载地址与散列保存在runs/alpaca-material。没有下载对应模型参数。项目自写的data/chinese-foundations.json为训练例子和虚构情境，不作亲历，也不能算独立验收。

## 2026-11-08 普通文章候选（未训练）

已核验0xDing/wikipedia-cn-20230720-filtered固定修订4cef256a3f426ae1d3f6930c8cd59a32d785d99d，524489333字节，SHA256为3a4dcb0e1b37436358d6cc5de28300071d55367395381958eba9fd38b3d78e32。提供方声明CC-BY-SA-3.0，基于中文维基2023-07-20快照，经筛选和简繁转换，共解析254547条。仅下载为后续文本候选，没有进入F对照训练。样本有公式丢失、残留标记与过时资料，不能直接视为可靠教材。原文及来源清单保留于runs/chinese-prose-material，不随源码上传；见[P23-prose-source-review.json](../reports/P23-prose-source-review.json)。
