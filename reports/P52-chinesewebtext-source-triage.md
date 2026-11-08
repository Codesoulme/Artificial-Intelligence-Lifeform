# P52：ChineseWebText 2.0 高质量子集初筛

固定源：[ChineseWebText2.0-HighQuality](https://huggingface.co/datasets/Morton-Li/ChineseWebText2.0-HighQuality)，版本 `8007a3ed1f6b9351f82afb2643f8bb6ff094b029`；它派生自 [CASIA-LM/ChineseWebText2.0](https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0)。来源数据卡声明 Apache-2.0；网页原始内容的具体权利仍需核实，不能将集合标注视为逐条授权保证。

只下载第一个 parquet 分片，127,629,302 字节、100,000 行，SHA-256 `b0713004c9664cbcb40613c51d742b0030bee0fbbdc774f21131f06f3ec222b1`，与远端文件元数据一致。数据卡声称保留 `quality_score >= 0.9`、`toxicity.score <= 0.01`，未做去重。采用固定种子 521109 对整个分片蓄水池抽样 30 条，查看来源字段、类别和每条的前 350 字；这是探索性检查，不是全量人工审查，也没有评估其统计误差。

样本中出现商业宣传、活动名单、时效性新闻、司法文书中的自然人信息、危险操作建议、断句或编码残留、多个无关句子拼接，以及古文与混合文体。领域标签也可能将非百科内容分类为百科。质量分和毒性分为自动预测，不代表事实可靠、适合作为助手答案或不含个人信息。

结论：**不得直接用于训练**。原始分片保存在本地 `runs/research/chinesewebtext2-hq/`，未加入 Git、未训练。下一步仅在不泄漏具体个人信息的前提下，构建来源可追踪的域分层筛选、个人信息过滤、近重复检测及与历史开发集的隔离审计；先抽样复核和进行小规模匹配实验，再决定是否扩容。对最终聊天验收需另设从未用于模型选择的独立题集。

整分片机器盘点见本地 `runs/research/chinesewebtext2-hq/inventory-v1.json`，脚本为 `scripts/audit_chinesewebtext_shard.py`，运行需要 `pyarrow`。100,000 行中，336 行匹配手机号样式，118 行匹配邮箱样式，122 行匹配身份证号码样式，4,661 行含司法相关词；这些是未经人工核实的规则命中，不是确证的个人信息数量，也不能证明其他行无隐私信息。规范化全文哈希仅发现 1 条重复行，该方法不能识别近重复。固定种子抽样仅保存行号、长度、类别、分数和文本摘要，不导出原文。

相关研究：[FineWeb 2 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)描述了按语言去重、筛选及多种数据处理消融；其结果不能直接推定本项目小模型的最佳配比。数据卡同时承认过滤可能漏掉有害内容及对不同语言覆盖不均。
