# P108：现代中文连续文本的新来源核查

调查日期：2026-11-09。只查询公开目录、数据卡及少量文件；未下载大型语料、训练、改模型或提交。结果均为线索，**本次确认可直接训练的新增量为零**。原始字节不是中文字符、词元或审核通过量。检索非穷尽。

## 判断

| 来源 | 固定版本和规模 | 权利及样本 | 决策 |
|---|---|---|---|
| [MicroPython 中文资料](https://github.com/shaoziyang/micropython-chinese-resources) | [固定树](https://api.github.com/repos/shaoziyang/micropython-chinese-resources/git/trees/32878347d789d58a956cc64ff6a4acb932f895fe?recursive=1)未截断；全树 `.md/.mdx/.rst` 552 文件、1,781,389 字节 | 根目录有 [CC BY 4.0 许可文本](https://raw.githubusercontent.com/shaoziyang/micropython-chinese-resources/32878347d789d58a956cc64ff6a4acb932f895fe/LICENSE)；[资料说明](https://raw.githubusercontent.com/shaoziyang/micropython-chinese-resources/32878347d789d58a956cc64ff6a4acb932f895fe/%E4%B8%AD%E6%96%87%E8%B5%84%E6%96%99%E8%AF%B4%E6%98%8E/readme.md)称收集、整理资料，[首页](https://raw.githubusercontent.com/shaoziyang/micropython-chinese-resources/32878347d789d58a956cc64ff6a4acb932f895fe/readme.md)称社区收集整理编写。许可文件未证明各文作者或翻译底稿均获授权。 | **优先作逐篇溯源，不可整包使用。** [machine 样本](https://raw.githubusercontent.com/shaoziyang/micropython-chinese-resources/32878347d789d58a956cc64ff6a4acb932f895fe/MicroPython%E5%BA%93/%E4%B8%93%E7%94%A8%E5%BA%93/machine/readme.md)约 97 KB，开头是中文解释，但有大量 API、代码、表格，来源和翻译权未核实。正文主题偏硬件，不能代替日常叙述。 |
| [Manim 中文文档](https://github.com/manim-kindergarten/manim_document_zh) | [固定树](https://api.github.com/repos/manim-kindergarten/manim_document_zh/git/trees/00f7656eadebf6da77cf2884981d22587bafbfaf?recursive=1)未截断；同口径 94 文件、188,916 字节 | [根许可文件](https://raw.githubusercontent.com/manim-kindergarten/manim_document_zh/00f7656eadebf6da77cf2884981d22587bafbfaf/LICENSE)为 CC BY 4.0，但[说明页](https://raw.githubusercontent.com/manim-kindergarten/manim_document_zh/00f7656eadebf6da77cf2884981d22587bafbfaf/README.md)的徽标链接指向 GPL 3.0，且称部分文档从其他分支或仓库构建。 | **许可范围有冲突，暂缓。** [安装问题样本](https://raw.githubusercontent.com/manim-kindergarten/manim_document_zh/00f7656eadebf6da77cf2884981d22587bafbfaf/cairo-backend/source/problems/v2.3.rst)主要是简短问答、命令、外部链接，连续性有限、可能过时。先查贡献与源文件许可，不应推断可训练或可商业使用。 |
| [科技爱好者周刊](https://github.com/ruanyf/weekly) | [固定树](https://api.github.com/repos/ruanyf/weekly/git/trees/c57527f3534e651337bcb6361b520df34ca8f4d2?recursive=1)未截断；同口径 430 文件、9,393,978 字节 | [固定说明](https://raw.githubusercontent.com/ruanyf/weekly/c57527f3534e651337bcb6361b520df34ca8f4d2/README.md)介绍周刊并邀请投稿；本次请求根 `LICENSE` 返回 404，GitHub 元数据也未识别许可。期目包含摘录、投稿和链接；**未发现适用于该仓库正文的开放训练或商业复用授权，不能因此断言不存在**。 | **仅作授权待查线索。** 标题和规模显示有长期中文写作，未经正文抽查及逐篇权利确认，不作质量、原创比例或可用规模判断。P92 的同作者 ES6 教程是不同作品，不重复计入。 |

统计范围包括导航、说明、标记、代码、英文及可能的重复。只抽看所链接少量文件，未按概率抽样；不估算通过率。根许可只能覆盖发布方有权许可的材料，不能替代他人作品、翻译和贡献者的权利证明。CC BY 4.0 通常允许商业分享及改编，但有署名、许可链接和修改标记等条件；训练、复制、分发与成果发布的适用义务须按具体方案判断。

## 大规模集合的排除证据

- [Chinese Debate Documents 数据卡](https://huggingface.co/datasets/DBWBD/Chinese_Debate_Documents)列 MIT、2,028 条、约 238.7 MB 数据量，称原始大学辩论音频经自动识别生成转写。数据整理者标示 MIT 不能证明辩手、赛事或录音权利人授权；含队伍、话题、发言人及时间信息，可能涉及个人信息，且识别错误会破坏连续性。**不下载整包，不计作已授权量。**
- [Chinese Patents 数据卡](https://huggingface.co/datasets/LiuHongwei1992/ChinesePatentsFrom1985to2025)明确排除直接商业利用、商业产品部署及未经许可的大规模专有模型训练；并称底层专利文本权利属于各自权利人。当前项目按非商业研究处理，但这些限制及底层权利仍须结合具体用途核实，不能据此断言研究用途必然被禁止或已经获准；专利文体也与日常聊天相距较远，暂不作为优先训练来源。
- [Chinese novel nonH 数据卡](https://huggingface.co/datasets/lainka0o0/chinese-novel-nonH-collect)只有 CC0 标签和宽泛规模档，未提供可核实的作品来源或作者授权链；[另一小说集合](https://huggingface.co/datasets/n-pn/chinese-novel-corpus)仅见 Apache 标签。**均不因标签而纳入**，规模档亦不代表可用小说数。

## 去重与隐私

对 P84–P107 的报告检索上述新仓库名及中文名称，未见它们作为同一来源评估；这不是全文级查重，也不证明外部汇编或已有训练数据未收录相同作品。OI Wiki 在 P92 已评估，因此本次不重复推荐。先按作品、原作与译作、固定版本和文段摘要核对已允许访问的旧材料；涉及受保护评测内容时只经授权隔离流程返回阻断结果，无流程则保持待定。

尚未完成隐私审核。教程也可能包含姓名、联系方式、账户、个人路径、示例凭证或截图；默认排除名单、讨论、联系方式及附件，发现敏感内容隔离整段。用于溯源的署名和审计材料与训练正文分开存放并限制访问。周刊还可能包含具名投稿、采访或招聘信息，不因公开而默认可复用。

## 可执行的下一步

1. **先解决权利再扩容。** 固定上述提交，为 MicroPython 按文件记录作者、底稿、贡献历史、原文和译文许可及第三方引用；按文件范围核实 Manim 徽标与许可文本的关系。对无法证明权利链的条目保持隔离。周刊只有在找到权利人覆盖拟议用途的明确许可后再继续，必要时另行取得书面授权；本次未联系任何人。
2. **限量抽查。** 分别冻结符合初步权利条件的内容文件清单，排除导航、代码、列表、联系方式及外部引用为主的文件；按长度、主题和原创/翻译线索分层，固定种子每源最多抽 10 个完整文件、总计不超过 1 MB。记录抽样概率、失败和截断；两名审阅者独立核对连贯性、权利、隐私、公式或代码依赖、时效与重复。没有合格清单则不抽正文；分层样本未经加权不能外推全库。
3. **先核算再决策。** 单列发现、可获取、权利通过、隐私与质量通过、去重后的完整中文字符和项目词表词元；保留来源、固定提交、路径、获取日期、许可证据和摘要。设技术主题及作品配额，防止少数项目主导。结果不足时继续寻找更多领域、权利人直接发布的文本；本次没有证明已找到可作为主体的大型通用来源。

## 获取边界

使用 GitHub 官方仓库检索、提交与树 API，以及 raw 子域固定单文件；未请求 GitHub robots 禁止的搜索或 tree 网页。GitHub 的 robots 建议使用 API；API 和 raw 子域的 robots 请求返回 404，不能作为版权授权。Hugging Face robots 返回 `Allow: /`，只读取公开检索元数据和 README，不请求数据分片或测试集。网络使用 `curl.exe --ssl-no-revoke`，保留证书链验证但未检查证书吊销。结果仅反映本次访问。
