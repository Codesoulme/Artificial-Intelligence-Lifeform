# P92：可扩展中文数据的非爬取来源调查

调查日期：2026-11-08。目标是扩大可追溯的中文训练候选，优先检查权利、出处、可下载形式和文档级信息。本次只访问项目说明、固定版本目录元数据及少量章节，不下载大型文件、不构建训练集、不训练。网络发现并非穷尽；下述条款是发布者的声明，不是每段文字的版权保证，也不是本项目训练或发布的专门授权。

## 结论

找到四个此前指定报告中未作为候选评估、实际可读固定源码的教育资料；其中 OI Wiki 的 465 个文档文件是本次规模最大的新增线索，但技术内容集中，包含代码、公式及引文，附加许可还须澄清。ES6 教程、Happy LLM 和 MIT 课程中文版都标有非商业限制；如果项目用途或成果发行涉及商业，不能把它们当成可自由商用的文本。**没有找到已证明兼具大规模、广泛现代中文、清楚逐篇权利及可直接训练的整包。** 当前均为未批准训练的调查候选；也没有找到符合上述条件的新型大规模中文对话集。

各行规模均按 GitHub 固定树中 blob 的 `size` 累加，不下载这些文件；它是 UTF-8 源文件字节，包括 Markdown、程序、公式、英文、链接及参考资料，不是中文字数、去重后数量或有效词元。样本只证明指定小文件在本次环境可读，不证明所有正文可获取或内容合格。

| 候选 | 固定版本、文件范围及声明规模 | 权利和筛选判断 |
|---|---|---|
| [OI Wiki](https://github.com/OI-wiki/OI-wiki/tree/c794414827bc5421fa1a463eb1a15ffc7941a1db) | 提交 `c794414827bc5421fa1a463eb1a15ffc7941a1db`；[完整目录](https://api.github.com/repos/OI-wiki/OI-wiki/git/trees/c794414827bc5421fa1a463eb1a15ffc7941a1db?recursive=1) `truncated: false`；`docs/` 下 `.md` 465 个、5,973,648 字节 | 非代码内容除另有注明外标为 CC BY-SA 4.0 **并附加** Star And Thank Author License；贡献者、外部引用、额外条件和共享义务需核实。技术教学规模相对较大，但不能当成通用会话语料。|
| [ES6 入门教程](https://github.com/ruanyf/es6tutorial/tree/d27a751258f25cb4d27de70161d3bcd7fe19251e) | 提交 `d27a751258f25cb4d27de70161d3bcd7fe19251e`；[目录](https://api.github.com/repos/ruanyf/es6tutorial/git/trees/d27a751258f25cb4d27de70161d3bcd7fe19251e?recursive=1) 未截断；`docs/` 下 `.md` 37 个、913,921 字节 | [固定说明](https://github.com/ruanyf/es6tutorial/blob/d27a751258f25cb4d27de70161d3bcd7fe19251e/README.md)称本书 CC BY-NC 4.0。需逐章核查贡献、引用和适用用途；代码及例子多。|
| [Happy LLM](https://github.com/DatawhaleChina/happy-llm/tree/1e0b3a1e6f8936b8ba4bbd0673e8e2227d196c96) | 提交 `1e0b3a1e6f8936b8ba4bbd0673e8e2227d196c96`；[目录](https://api.github.com/repos/DatawhaleChina/happy-llm/git/trees/1e0b3a1e6f8936b8ba4bbd0673e8e2227d196c96?recursive=1) 未截断；全树 `.md` 28 个、788,692 字节 | [固定说明](https://github.com/DatawhaleChina/happy-llm/blob/1e0b3a1e6f8936b8ba4bbd0673e8e2227d196c96/README.md)声明作品 CC BY-NC-SA 4.0。统计包含英文说明、附加章节和未完成内容；专业性强，需核外部材料、贡献和相同方式共享义务。|
| [计算机教育中缺失的一课中文版](https://github.com/missing-semester-cn/missing-semester-cn.github.io/tree/4f2fedde4f519c0466670b055efdd76af248574e) | 提交 `4f2fedde4f519c0466670b055efdd76af248574e`；[目录](https://api.github.com/repos/missing-semester-cn/missing-semester-cn.github.io/git/trees/4f2fedde4f519c0466670b055efdd76af248574e?recursive=1) 未截断；仅 `_2026/` 下 `.md` 10 个、235,044 字节 | [固定说明](https://github.com/missing-semester-cn/missing-semester-cn.github.io/blob/4f2fedde4f519c0466670b055efdd76af248574e/README.md)称课程内容 CC BY-NC-SA 4.0，并链接原课程的翻译说明；应核对英文原文、译者和具体文件的授权链。仅计 2026 目录，不代表全部中文课程规模。|

## 核验和可获取性

1. **OI Wiki。** 项目[固定版权说明](https://github.com/OI-wiki/OI-wiki/blob/c794414827bc5421fa1a463eb1a15ffc7941a1db/README.md)将非代码内容标为 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 和附加的 [Star And Thank Author License](https://github.com/zTrix/sata-license)，并说使用时应为仓库点赞；其同一说明又称分享时没有附加限制。不要擅自忽略附加条款或认定其与 CC 条款的关系已经解决；本次读取该附加项目的 `LICENSE` 原始地址失败，未核实其全文，须向维护者确认具体适用性。实际读取[均摊分析章节](https://raw.githubusercontent.com/OI-wiki/OI-wiki/c794414827bc5421fa1a463eb1a15ffc7941a1db/docs/basic/amortized-analysis.md)，开头是中文解释，也包含代码术语和公式。页面结构、参考来源、单页特别许可及贡献记录需逐篇核对； Git 历史与文件路径可作为贡献线索，不能自动得到完整署名名单。
2. **ES6。** [数组章节源码](https://raw.githubusercontent.com/ruanyf/es6tutorial/d27a751258f25cb4d27de70161d3bcd7fe19251e/docs/array.md)实际可读，有中文解释和大量程序示例。[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)含署名、非商业和标明修改要求。说明称纸质第三版于 2017 年出版、网站持续修订；本次只研究固定源码，不能把网页声明套到出版社排版或其他版本。文件数包含致谢等非正文，不能当作章节合格数。
3. **Happy LLM。** [第一章源码](https://raw.githubusercontent.com/DatawhaleChina/happy-llm/1e0b3a1e6f8936b8ba4bbd0673e8e2227d196c96/docs/chapter1/%E7%AC%AC%E4%B8%80%E7%AB%A0%20NLP%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5.md)返回连续中文教学文本；它也有技术概念和可能过时的能力判断。文件树包括附加章节、`README_en.md`、未完成的内容和多份简短说明，故 28 个文件不是 28 篇完整教学文章。[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)的非商业及共享条件需按实际用法评估；章节中引文、图片、示例与各参与者权利不能由仓库标签代替。
4. **课程中文版。** [2026 讲义示例](https://raw.githubusercontent.com/missing-semester-cn/missing-semester-cn.github.io/4f2fedde4f519c0466670b055efdd76af248574e/_2026/agentic-coding.md)实际可读，含标题、日期、状态、中文说明和程序片段。翻译团队说明链接到[原课程许可及贡献说明](https://missing.csail.mit.edu/license)；本次未逐条证明英文底稿与各译稿的对应版本，亦未核实所有讲义译者的权利。按讲义和语言版本保留关系，不能将同文英中两版当成独立原创材料。

上述每个仓库可通过固定树 API 得到路径、Git blob 标识和字节规模，再通过固定提交的 `raw.githubusercontent.com` 地址获取选定小文件；也可通过仓库固定版本页面选择发布包。API、源码和页面可访问性分别核实；本次实际取得的是目录、说明及列举的小文件，未下载整包。元数据不是逐篇作者和许可清单：后续须从文件历史、书目、正文和明确的权利声明补充，不能仅凭路径推断作者。

## 与既有调查的关系

已检查 P52、P59 和 P68–P91 对应报告的清单及来源相关内容，并对这些报告检索四个项目名称及仓库名称；未发现上述四项被作为候选评估。P52、P59 的网页汇编，以及 P68 文库、P70 语音、P71–P72 百科、P73 Aya、P74–P75 WildChat、P78 AISHELL、P84 教科书与故事、P85 新闻和论文索引、P86 新闻、P87 数学书与课程、P88 Linux 101、P89–P90 科普、P90 KdConv 和 P91 近代作品不在此重复充当新发现。P87 的《动手学深度学习》与 Happy LLM 都是机器学习教材，但不是同一项目；可能共享背景内容或引用，需要实测去重。检索项目名没有出现并不证明任何一段文字未被旧来源收录。

这些候选通过发布者提供的 Git 源码及目录获得，不要求项目自行进行网页爬取；不代表发布者的每条材料都没有外部来源。现有报告中对对话与隐私已有较充分的反例，四项均非对话数据；本次不以技术说明替代自然交流覆盖，也不虚报可用对话规模。

## 建议的可信筛选流程

1. **权利先行。** 先确定项目及成果的具体用途，逐源记录固定提交、原始说明、适用许可证、非商业与共享要求、额外条款和例外文件；对 OI Wiki 的附加条件、各仓库第三方引文及课程译文权利链取得明确证据，未知或用途不符者隔离。许可标签、开放仓库、可访问和作者拥有全部权利是不同命题。必要时直接向权利人核实；记录答复，不由抽样推定全库授权。
2. **构建文档账本。** 后续经允许后只按需取得文件，记录仓库、提交、路径、blob、实际字节摘要、标题、作者或贡献证据、原作和译文关系、源链接、许可、时间、第三方片段、提取版本和排除原因。源说明和每章特殊标注分别保存；不能定位贡献者、来源或条款的内容暂缓。先处理小规模样本，确认可行再扩大。
3. **防止重复与泄漏。** 先按整书、章节族、翻译及不同版本分组，再在受保护的流程内与既有训练源和开发、最终验收材料做精确及字符片段近重复检查；相似疑似项整组隔离并复核。仅把摘要和阻断结果交给研究者，避免浏览受保护文本。相似度方法不能保证发现所有语义改写或祖先模型曾见材料。
4. **先冻结质量规则再独立抽查。** 在允许的训练侧，按仓库、主题、章节和长度分层，先冻结文档清单、版本、样本种子、权利门槛与质量表，再由独立审查者完整阅读样本并对照原文；统计通过、存疑、排除及各原因。检查现代中文连贯性、公式或代码删除后是否仍完整、事实时效、个人信息、版权引用、重复以及文本抽取损伤。对技术性、图表依赖、非正文和未完稿单列；不能用中文占比或章节名代替判断。样本量和抽样设计决定可作的推断，分层均衡样本的通过率不能直接当总体比例。
5. **单独计量。** 分别报告发现量、可获取量、权利审查通过量、正文完整且质量合格量、去重后中文字符数与使用项目固定词表统计的词元量；再记录来源及作品组配额。有效训练量须排除代码、标记和重复暴露，并结合项目任务核查；原始字节不可直接换算。即使通过初审，训练方案、发布义务和最终评估仍需单独确定。

## 访问说明与边界

使用 Windows `curl.exe --ssl-no-revoke` 访问 GitHub API 和固定源码；保留证书链校验，但未完成证书吊销检查。目录统计只依据返回的固定树元数据，四个树均标记 `truncated: false`；未逐文件获取或重算内容摘要。具体响应随网络环境变化。只抽读四个指定章节的开头，未开展全文质量评审或训练权利的法律判定；附加许可原始文件请求失败也不代表该条款不存在或永久不可访问。本次未取得大规模中文对话、未下载大文件、未训练。
