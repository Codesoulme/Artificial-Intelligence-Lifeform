# P68：中文维基文库首分片探索性审查

候选为 [Wikimedia Wikisource 中文 20231201](https://huggingface.co/datasets/wikimedia/wikisource/tree/f31a033f5f3d2107b3e864e578710df104a00baa/20231201.zh)，固定版本 `f31a033f5f3d2107b3e864e578710df104a00baa`。数据卡标注 CC BY-SA 3.0 和 GFDL，作品与现代整理、翻译的具体权利及署名要求需继续核对；集合标签不等于逐篇许可证明。首分片 `train-00000-of-00011.parquet` 下载 169,190,012 字节，SHA-256 `7da28a2c44429d029680346f4e700c5b30c2545a9a47e783a732268cd4dfc1a4`，与固定版本的远端 LFS 元数据一致；包含 24,152 行，字段为 id、url、title、text。下载时系统证书链校验启用，但因本机吊销查询不可用，使用 curl `--ssl-no-revoke`；这降低了吊销检查的保障。数据仅存本地 `runs/research/wikisource-zh/`，未用于训练。

以 Python `random.Random(681109).sample(range(24152), 30)` 抽取 30 行，检查题名及每篇前 220 字，并非全文审查。样本包括古代史书、诗歌、小说、法律条文、目录、历史医书及现代科普文章；大量使用繁体或文言。还见残缺标记与未转换的维基语言标签。医疗内容、过时法规及历史陈述不应不加区分当作当前建议或事实。样本仅来自首个分片，且顺序分片不能代表全体，不能由此估计整个集合的现代白话比例。

结论：该集合具有来源追溯价值，但首分片抽样与当前现代中文聊天训练目标不匹配，**未批准整片训练**。若未来使用，先按作品和版本分层、核对版权与文本质量、隔离医疗与时效性材料，并排查与旧语料及评测集近重复；不能把古文规模直接计为可用现代聊天训练量。

并行调查还发现 [中文古籍语料](https://huggingface.co/datasets/gujilab/chinese-classical-corpus) 和 [Project Gutenberg 中文目录](https://www.gutenberg.org/browse/languages/zh)，均以古典作品为主，集合声明不自动证明现代整理版本的权利；它们未下载或批准。新版中文维基百科与项目已有版本可能高度重合。后续仍需寻找来源清楚且更符合现代口语的连续文本与多轮对话。
