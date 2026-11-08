# P97：现代中文连续文本来源补充调查

调查日期：2026-11-08。目标是寻找机构发布、可实际读取、有明确复用条件的现代中文教材、儿童读物及科普文章。本次只读取少量页面、固定版本的四个章节文件与仓库树元数据；没有下载大规模数据、训练或构建语料。候选均未经过逐篇权利和质量审查。

## 结论

**没有找到同时证明符合“大型机构、多作者、现代中文原生叙述、数万字以上、权利链清楚且适合批量复用”的新来源。** 新查到 Datawhale 社区的两本可读中文教材，均为 CC BY-NC-SA 4.0 声明，但领域高度集中于机器学习、公式密集，且各有第三方材料问题。它是开源学习社区，不应称为大型出版机构。尤其《蘑菇书》前三章是根据外部课程整理，关联上游课程写明仅供个人教育用途，须厘清权利后才可考虑复用；它不能因项目声明而被认定为权利已解决。两书都是技术说明文，不是儿童故事或广泛的日常叙述。

| 候选 | 实际读取与可验证规模 | 许可及判断 |
|---|---|---|
| [《蘑菇书 EasyRL》](https://github.com/datawhalechina/easy-rl) | 固定版前三章实际读到 38,280 个基本汉字，含标题、图注及标记内汉字；固定目录选出 15 个章文件、420,044 字节。后者不是正文汉字数。 | 项目标 CC BY-NC-SA 4.0；所读章节按项目说明是根据其他课程整理，上游权利尚未厘清。**暂不用于批量复用**。 |
| [《南瓜书》](https://github.com/datawhalechina/pumpkin-book) | 固定版第一章可读，4,415 个基本汉字；固定目录选出 16 个章文件、671,194 字节。尚未证明正文达到数万字。 | 项目标 CC BY-NC-SA 4.0；作为《机器学习》伴读与公式解析，原书引文和图表等权利需要核验。**仅列待审线索**。 |

## 《蘑菇书》：可读、篇幅可量，但上游条件是实质障碍

固定提交为 `384edfea97870facddc43d7110f2fb398f528cb4`。项目的[固定说明](https://raw.githubusercontent.com/datawhalechina/easy-rl/384edfea97870facddc43d7110f2fb398f528cb4/docs/README.md)介绍其为中文强化学习教程，列出多位贡献者及不同分工；贡献者人数不等于每章都有多位作者，也不能证明所有章节权利完整。说明称纸本经过出版社编辑，因此本调查只涉及仓库固定源码，不把其声明扩展到纸本。

实际读取[第一章](https://raw.githubusercontent.com/datawhalechina/easy-rl/384edfea97870facddc43d7110f2fb398f528cb4/docs/chapter1/chapter1.md)、[第二章](https://raw.githubusercontent.com/datawhalechina/easy-rl/384edfea97870facddc43d7110f2fb398f528cb4/docs/chapter2/chapter2.md)及[第三章](https://raw.githubusercontent.com/datawhalechina/easy-rl/384edfea97870facddc43d7110f2fb398f528cb4/docs/chapter3/chapter3.md)。三者分别有 13,929、13,880、10,471 个 U+4E00–U+9FFF 字符，共 **38,280**；这是对完整 Markdown 字符串的机械计数，包括标题、图注和公式等位置的中文，未去重、未做正文提取、未作质量验收，不能等同可用中文字数。第一章介绍游戏中的延迟反馈，第三章以遇熊来说明状态转换；后者只是抽象示例，不可当作野外安全建议。

[固定完整目录](https://api.github.com/repos/datawhalechina/easy-rl/git/trees/384edfea97870facddc43d7110f2fb398f528cb4?recursive=1)显示 `truncated: false`。仅对路径完全匹配 `docs/chapter[0-9]+/chapter[0-9]+.md` 的 blob 累加其 `size`，得 15 个、420,044 字节；其中第 15 章仅 336 字节，且该规则不含第 14 章的不同命名文件，所以不能将 15 当作 15 篇完整正文，也不能换算成中文字数。其他章节只核对目录，并未读取其全部内容。

[项目说明](https://raw.githubusercontent.com/datawhalechina/easy-rl/384edfea97870facddc43d7110f2fb398f528cb4/docs/README.md)和[许可文件](https://raw.githubusercontent.com/datawhalechina/easy-rl/384edfea97870facddc43d7110f2fb398f528cb4/LICENSE)标示 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)：包含署名、非商业和相同方式共享等条件；项目的实际用途及产物须另行判断。更重要的是，说明明确称第一、二章根据周博磊《强化学习纲要》整理，第三、十二章根据其他课程整理，第四至十一章来自李宏毅课程的部分内容。上游[《强化学习纲要》仓库说明](https://raw.githubusercontent.com/zhoubolei/introRL/master/README.md)写明课程仅供个人教育用途；该链接是可变分支，需固定版本并核对相应授课内容。该说明不自动决定整理稿各段权利，也不能证明整理者取得了复用授权。应向相关权利人核实整理稿、视频讲解、讲义、图表及作者贡献之间的授权关系；在此之前，本报告不把三章计入已授权可用量。中文授课和中文整理亦不证明每一段均为独立原生创作。

## 《南瓜书》：伴读教材，许可范围需拆开核查

固定提交 `58add39800dfbef65c93b2226d9a514f4cc9da0e` 的[项目说明](https://raw.githubusercontent.com/datawhalechina/pumpkin-book/58add39800dfbef65c93b2226d9a514f4cc9da0e/README.md)称这是对《机器学习》相关公式的解释和推导，感谢多名早期贡献者，说明作品采用 CC BY-NC-SA 4.0。[第一章](https://raw.githubusercontent.com/datawhalechina/pumpkin-book/58add39800dfbef65c93b2226d9a514f4cc9da0e/docs/chapter1/chapter1.md)实际可读，使用中文连续说明算法、模型及样本，但夹杂数学表达和上游书籍引用。按同一基本汉字范围计数为 **4,415**，是含 Markdown 的文件计数，不是经筛选的正文。

[固定树](https://api.github.com/repos/datawhalechina/pumpkin-book/git/trees/58add39800dfbef65c93b2226d9a514f4cc9da0e?recursive=1) `truncated: false`；按上一节相同路径规则筛选，共 **16 个章文件、671,194 字节**，排除其他命名文件和法文文件。该数含数学、标记、英文、链接，且包含仅 4,032 字节的第 15 章；仅凭它无法证明中文有效正文超过数万字，也无法证明章节完整或各章均可获取。

这是围绕既有书的伴读作品，项目许可不能授予原书原文、图像、公式排版或第三方引文的权利。需先区分独立说明与借用内容，核实原书及贡献者授权，并评估非商业和相同方式共享条件。样章支持存在中文写作，但不能仅凭一章认定全书均为原创中文。教材专业、内容依赖原书上下文，删除公式和图后也可能丢失论证；不宜当作叙事语料的直接替代。

## 排除与访问边界

- 已对 P84–P96 的报告检索两本书的中文名及仓库名，未见作为来源评估；这不证明内容与项目现有文件不重合。维基项目、Storybooks Canada、StoryWeaver、Frontiers 中文文章和 P92 的 Happy LLM 等均不重复计入新发现。
- [教育百科 robots](https://pedia.cloud.edu.tw/robots.txt)允许词条路径，但禁止搜索、接口及教材等指定路径；首页请求中途连接重置，本次未核实词条正文许可或连续文本规模，也未进入禁区。允许索引本身不等于允许批量复用。
- 《科学月刊》站点的 robots 请求遇到浏览器验证，未继续访问或绕过，因此无法判断该站许可与正文规模。[Open University robots](https://www.open.edu/robots.txt)同样遇到验证，未作来源结论。Bing 的 robots 对通用爬虫禁止 `/search`，没有请求其搜索页。
- [GitHub robots](https://github.com/robots.txt)建议使用官方接口；本次没有抓取其禁止的目录页面，所用 GitHub API 与 raw 子域的 `/robots.txt` 均返回 404。仅使用公开仓库的官方树元数据和少量单文件，未使用受限接口；404 不代表对作品版权的授权。联网使用 `curl.exe --ssl-no-revoke`，保留证书链验证，但未完成证书吊销检查。

后续若要推进，应先以章节为单位保存原文版本、作者与来源，解决上游许可，并核对计划用途与 NC/SA 条件；再按完整段落审查公式依赖、质量及重复。当前不能将文件字节、目录数量或样章中的机械字符数称为已审核的可用规模。
