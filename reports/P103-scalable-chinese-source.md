# P103：可扩展现代中文连续文本来源调查

调查日期：2026-11-09。范围是公开来源发现、固定版本目录、许可声明与少量样章；没有下载大规模正文、构建语料、读取最终测试、修改模型或运行训练。下述来源都只是待审线索，不是训练批准。网络调查并非穷尽。

## 结论与优先级

**本次没有找到经证明可直接用于大规模、通用现代中文连续预训练的整包。** 新找到三组可固定版本的中文软件教程，合计约 189 万 UTF-8 源文件字节，但包含代码、标记、英文和重复材料，不能换算成有效中文或词元；主题集中，也不是自然对话。更有扩展性的路线是建立可追溯的**多来源发现和逐文档筛选流程**，而不是将单个仓库充当主体。此次已核实的合格训练量为零；这是尚未完成审核，不是说这些材料没有价值。

| 优先级 | 固定来源及目录规模 | 许可证据 | 现实限制 |
| --- | --- | --- | --- |
| 1 | [Zotero 中文文档](https://api.github.com/repos/zotero-chinese/wiki/git/trees/1d2e1ec6eef8be65c31b809a94feb9b81a009485?recursive=1)：全树 `.md/.mdx/.rst` 99 个，791,007 字节 | [固定说明](https://raw.githubusercontent.com/zotero-chinese/wiki/1d2e1ec6eef8be65c31b809a94feb9b81a009485/README.md)称文档 CC BY-SA 4.0、代码 MIT | 明确含译文；[样章](https://raw.githubusercontent.com/zotero-chinese/wiki/1d2e1ec6eef8be65c31b809a94feb9b81a009485/user-guide/why-use-zotero.md)标明部分翻译自官方站，须核底稿、译者与翻译权；文献管理主题窄，操作与价格易过时。|
| 2 | [清华移动应用开发 Flutter 讲稿仓库](https://api.github.com/repos/thu-mobile-dev/book/git/trees/d3c81525e3e5a2f1b30ff08829c116e4684fd839?recursive=1)：同一扩展名范围 41 个，301,989 字节 | 仓库有 [CC BY 4.0 许可文件](https://raw.githubusercontent.com/thu-mobile-dev/book/d3c81525e3e5a2f1b30ff08829c116e4684fd839/LICENSE)；[说明](https://raw.githubusercontent.com/thu-mobile-dev/book/d3c81525e3e5a2f1b30ff08829c116e4684fd839/README.md)称为课程讲稿 | 许可文件不证明所有第三方内容及各作者权利；[样章](https://raw.githubusercontent.com/thu-mobile-dev/book/d3c81525e3e5a2f1b30ff08829c116e4684fd839/docs/LECTURE/1/main-todoapp.md)是中文说明与大量 Dart 程序交织，去掉代码可能破坏上下文。仓库名不能证明全部内容由学校持有权利。|
| 3 | [小白 Python 教程](https://api.github.com/repos/walter201230/Python/git/trees/bc8bb4579234dcafa0d179d1ff048410a8d06467?recursive=1)：同一扩展名范围 102 个，798,079 字节 | 有 [CC BY 4.0 许可文件](https://raw.githubusercontent.com/walter201230/Python/bc8bb4579234dcafa0d179d1ff048410a8d06467/LICENSE)，适用范围及第三方材料仍需确认 | [样章](https://raw.githubusercontent.com/walter201230/Python/bc8bb4579234dcafa0d179d1ff048410a8d06467/Article/PythonBasis/python0/WhyStudyPython.md)含旧排名、薪资数据和具名新闻引文；[说明](https://raw.githubusercontent.com/walter201230/Python/bc8bb4579234dcafa0d179d1ff048410a8d06467/README.md)称 2026 年使用生成工具全面更新教程，范围和人工核实情况不明。需逐章查引文权利与事实，不宜视为全为人类原创。|

三个表中目录的 `truncated` 均为 `false`；筛选条件是整个返回树中 `type=blob` 且路径不区分大小写以 `.md`、`.mdx` 或 `.rst` 结尾，累加 API 的 `size`。因此数字包括说明、贡献指南、目录、图注和源码，不是正文、文件去重数或质量通过量；不保证其他格式不存在。样章只读了所列少量文件，未完成全库审查。许可、可获取、隐私和文本质量是独立门槛；CC BY 的署名、许可链接和修改标示及 CC BY-SA 的相应共享条件，需结合具体复制、训练和成果发布方式评估。集合声明不能替第三方引文或翻译底稿授权。

同一检索还发现 [SRE 工程师指南](https://api.github.com/repos/sunface/SRE-course/git/trees/e042ed65bf803782d8de2d5ca9c8deb9e15708fd?recursive=1)：固定树只有 2 个上述扩展名文件、1,647 字节，不值得以仓库受欢迎程度推断其文本规模；不列为规模候选。

## 与 P84–P97 的关系

本次先核对已有来源审计，特别是 [P84](P84-continuous-source-search.md)、[P85](P85-continuous-source-research.md)、[P86](P86-wikinews-triage.md)、[P87](P87-continuous-source-discovery.md)、[P88](P88-linux101-local-research.md)、[P89](P89-general-chinese-sources.md)、[P90](P90-frontiers-zh-bounded.md)、[P91](P91-modern-public-domain-discovery.md)、[P92](P92-scalable-chinese-data-search.md)、[P94](P94-data-quality-research.md)、[P95](P95-storyweaver-investigation.md)、[P96](P96-reviewed-story-candidate.md) 和 [P97](P97-chinese-continuous-data-sources.md)。P93、P94 其他报告主要关乎对照试验，P95 另有故事重合审计；未将其当作新的来源。对 P84–P97 报告按上述三个仓库名、Python 仓库作者及 SRE 仓库名检索，未发现其被作为来源评估；这不证明它们的内容未被旧语料或外部集合收录。

已有结果约束排序：P86 新闻 40 篇抽审仅 3 篇初判受限候选，不能凭转储大小推断可用量；P84、P96 的儿童故事通过初审部分很小；P87、P92、P97 发现的教材也偏技术且有权利或非商业条件。P85 的 FineWeb 2 和开放论文索引只能发现来源，集合许可和索引标签不能代替网页原作者、版本或全文许可。这里不重新把这些数据计入增量，也不建议整包摄取。

## 隐私与质量

课程正文一般不像对话日志那样天然包含完整私人交流，但仓库可能出现作者或学生姓名、邮箱、帐号、示例凭证、个人路径、截图文字及问题记录；没有实际审查不能声称不存在。只以内容文件为候选，默认排除贡献者名单、联系方式、讨论记录、截图和凭证，敏感例子整段隔离而非仅做正则替换。对允许保留的署名和审计信息单独访问控制，不将作者资料自动并入正文。

按文章而不是全库判断自然中文、文段完整性、版本时效、第三方引用、代码和图片依赖；事实性内容尤其是安装命令、薪资和软件版本要标记日期。检查机械生成或改写痕迹，但不以检测器分数判定作者身份。对教程施加主题与作品配额，避免大量重复步骤冲淡叙事、对话及普通生活题材。三个样本都不足以估计总体通过率或证明任何来源适合作为主料。

## 可直接执行的下一步

1. **先定权利边界。** 为每个固定提交建立来源清单，记录项目与文件的许可声明、版本、作者及贡献记录、底稿、引文、图片和具体用途。优先审查 Zotero 的独立中文页面；对明确译文先核上游许可和翻译授权，对 Python 新闻引文单独隔离，对 Flutter 讲稿确认贡献者和第三方示例。保存证据链接与获取时间；权利来源不明、用途与条件不符或版本无法锁定即停。
2. **做小样本，不下载整库。** 从每个固定树中排除 README、导航、贡献名单及非内容路径后，先冻结可审查文件列表；按来源、文件长度和原创/译文线索分层，固定随机种子各抽最多 10 篇，不足则全抽。仅在 robots 和站点条款允许时获取选定文件；设置总量上限 30 篇、2 MB，超限先缩减，记录下载失败及抽样概率。由两名审阅者独立检查完整文章，记录通过、存疑、排除与原因；样本前缀不能代表全文。分层样本的未加权通过率不作全库估计。
3. **建立可扩展来源账本。** 对合格样本记录固定版本、文件路径和 blob、实际原始字节摘要、作者/译者、底稿、许可、第三方内容、日期、提取方式、隐私和审核结论。之后可利用 GitHub 官方仓库检索与树接口，按中文教程、公开教材等多个领域扩展候选；搜索结果及许可标签只用于发现，再逐仓库核对正文和权利。每次限定请求量，检查响应状态、目录是否截断和新版本变更；不得将大量相似仓库计为独立作品。
4. **先隔离重复，再估算增量。** 在训练侧候选内部以及已明确允许检查的历史训练材料中，以作品族、原文/译文、精确摘要和字符片段查重，近重复交人工审查。记录检查范围与漏检限制；本任务没有读取最终测试，后续涉及受保护评测材料应由被授权的隔离流程执行，仅返回阻断结果，不向来源研究者暴露内容；若无此流程则保持待定。P95 的重合结论只覆盖其列出的历史文件，不能泛化。
5. **统计漏斗并设停点。** 分别报告发现的文件数和源字节、可读取数、权利通过数、隐私和质量通过数、去重后的完整中文字符与使用项目固定词表得到的词元；不能把仓库体积或样章比例直接外推。若不足以满足既定目标，应继续寻找不同领域且可核权的发行方，或在另行获准后寻求权利人书面授权；本次没有联系任何人，也没有启动训练。

## 访问与方法边界

先读取 [GitHub robots](https://github.com/robots.txt)：其要求相应爬虫避免搜索页及若干目录路径，并建议使用官方 API；本次采用 `api.github.com` 的公开仓库检索、提交及树接口和 `raw.githubusercontent.com` 的固定单文件，未自动请求 GitHub 被禁止的搜索或目录页面。两子域的 `/robots.txt` 此次均返回 404；这不是许可证明。来源检索使用 GitHub 仓库 API 的许可和中文教程关键词，结果受索引、排序和关键词限制。访问使用系统 `curl.exe --ssl-no-revoke`：证书链校验保留，吊销状态未检查；Python 默认证书链请求失败后未使用关闭证书验证的方式。本文链接固定提交和原始文件，网络状态仅反映此次调查。
