# P87：现代中文连续文本的补充来源调查

调查日期：2026-11-08。目标是寻找不同于 [P84](P84-continuous-source-search.md)、[P85](P85-continuous-source-research.md) 和 [P86](P86-wikinews-triage.md) 的机构讲义、开放教材及有明确权利声明的书目。只读取网页、少量单文件和仓库目录元数据；没有下载大型正文包、训练或批准任何来源用于训练。网页和接口的可访问性仅代表本次环境。调查不是穷尽式检索。

**结论：找到三组可实际读取、能固定版本的中文教材，其中两卷正式出版过的数学书对正文有作者署名和 CC BY 4.0 声明；大学课程讲义及另一部教材则有 CC BY-SA 4.0 的站点或项目声明。它们都偏技术，不能充当广泛现代中文的主体。** 文化机构目录及适用条款可读，所试的三个正文接口未能取回内容，暂不能核验具体文章、数量或逐条权利。各项均为调查候选，不是训练许可结论。

| 来源 | 本次可确认的固定文本和规模 | 权利证据与定位 |
|---|---|---|
| 李文威《代数学方法》卷一、卷二 | 两个固定源码版本分别含 10、9 个章节文件，合计约 **295.8 万字节**，含 TeX 标记、公式及非汉字；抽读了两卷首章 | 作者在各卷主文件和首章直接声明 CC BY 4.0；出版社版本与第三方引用需另核。**书目候选** |
| 中科大 Linux 用户协会《Linux 101》 | 固定目录的 `docs/` 中 **35 个 Markdown 文件、484,174 字节**；实际读到第一章中文 | 站点及项目说明称文档采用 CC BY-SA 4.0；需按贡献者、引用和第三方材料核查。**机构课程候选** |
| 《动手学深度学习》中文版第二版 | 固定目录中 `chapter_` 下排除 `_origin.md` 后 **140 个 Markdown 文件、2,133,601 字节**；读到引言 | 固定构建配置列 CC BY-SA 4.0 和 MIT-0，另有 Apache 2.0 仓库许可文本，适用边界需确认；中文版含译作。**待澄清候选** |
| 台湾文化机构开放资源 | 原站目录实际列出记忆库、传统工艺及建筑导览；三个内容接口本次未返回正文 | 各资源页标注政府资料开放授权条款第 1 版，但具体记录、引用及访问均未核。**仅为线索** |

上述字节数来自 GitHub 对固定树中文件大小的元数据相加，不是汉字数、有效词元数或质量合格量，也不是正文下载量；公式、程序、标记、练习、引用及重复内容均未剔除。不同项目的用途、结构和主题不同，不应直接比较字节得出训练价值。

## 1. 《代数学方法》：作者署名的正式教材

**原站与版本。** 李文威的源码仓库分别提供[卷一](https://github.com/wenweili/AlJabr-1)和[卷二](https://github.com/wenweili/AlJabr-2)。卷一 [README](https://github.com/wenweili/AlJabr-1/blob/1e2d0bf4e5b71194b85958ca6b16a6f396ad25be/README.md)称 2019 年由高等教育出版社出版，ISBN 978-7-04-050725-6，该源码为 2023 年修订版；卷二 [README](https://github.com/wenweili/AlJabr-2/blob/d56e07ebea5d29dceeaf9d227ae81c00eb720e70/README.md)称 2024 年由同一出版社出版，ISBN 978-7-04-062754-1。这里考察的是作者公开的源码版本，不能自动将声明扩大到出版社排版、其他版本或其他人的附件。

**实际取得的固定文本。** 卷一提交 `1e2d0bf4e5b71194b85958ca6b16a6f396ad25be` 的[第一章原始文本](https://raw.githubusercontent.com/wenweili/AlJabr-1/1e2d0bf4e5b71194b85958ca6b16a6f396ad25be/chapter1.tex)和卷二提交 `d56e07ebea5d29dceeaf9d227ae81c00eb720e70` 的[第一章原始文本](https://raw.githubusercontent.com/wenweili/AlJabr-2/d56e07ebea5d29dceeaf9d227ae81c00eb720e70/chapter1.tex)均返回 200，包含现代中文说明、公式及参考引用。固定版本还列有独立 PDF，但本次未下载；不把 PDF 视作已取得或已与源码逐页比对。

**授权链。** 卷一[主文件](https://raw.githubusercontent.com/wenweili/AlJabr-1/1e2d0bf4e5b71194b85958ca6b16a6f396ad25be/Al-jabr-1.tex)和首章写明 Copyright 2018 李文威及 CC BY 4.0；卷二[主文件](https://raw.githubusercontent.com/wenweili/AlJabr-2/d56e07ebea5d29dceeaf9d227ae81c00eb720e70/Al-jabr-2.tex)和首章写明 Copyright 2024 李文威及相同版本，明确许可复制、分发和修改。[CC BY 4.0 官方条款](https://creativecommons.org/licenses/by/4.0/)包含署名、许可链接及修改标示等要求。作者源码中的声明比目录标签更接近作品，但并未逐一审核引用的原文、图、贡献或出版社合同；尤其首章实际存在引文，不能默认第三方内容均在作者许可范围内。该许可也不是对本项目训练与发布方案的专门说明。

**规模和限制。** [卷一文件目录](https://api.github.com/repos/wenweili/AlJabr-1/contents?ref=1e2d0bf4e5b71194b85958ca6b16a6f396ad25be)的 `chapter1.tex` 至 `chapter10.tex` 共 1,174,359 字节；[卷二文件目录](https://api.github.com/repos/wenweili/AlJabr-2/contents?ref=d56e07ebea5d29dceeaf9d227ae81c00eb720e70)的 `chapter1.tex` 至 `chapter9.tex` 共 1,783,650 字节。该统计不含导言、附录、索引、图片和 PDF，亦未去掉公式和 TeX。两卷论述高度专业，符号、定理、索引与交叉引用密集；简单删除公式会破坏句意，卷二与卷一也有概念重述，须分段审核、去重并保留上下文。

## 2. 中科大 Linux 用户协会《Linux 101》

**原站与实际文本。** [课程教材站](https://101.lug.ustc.edu.cn/)介绍其为 Linux 101 活动的配套书，正文共九章及附录，由 LUG@USTC 发布并维护功劳簿。固定提交 `4099db4a1f90e8e89cac3eca4bf2e36b2932e3f1` 的[第一章源码](https://raw.githubusercontent.com/ustclug/Linux101-docs/4099db4a1f90e8e89cac3eca4bf2e36b2932e3f1/docs/Ch01/index.md)实际返回中文导言及连续说明。在线站可能随时间变化；固定源码才是这里的审查对象，不声称线上当前页面与该提交逐字一致。

**许可与权利链。** 站点说明所有内容以 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 开放；固定版本的[项目说明](https://github.com/ustclug/Linux101-docs/blob/4099db4a1f90e8e89cac3eca4bf2e36b2932e3f1/README.md)同样明确文档以该协议发布，并链接仓库许可。此为站点和维护者对文档的声明；具体贡献者、引用来源、截图、代码块和嵌入的外部作品仍须单独核验。该版本的署名及相同方式共享条件应结合拟议处理、再分发和产物性质评估，不能假定对模型或衍生成果没有影响。

**规模与限制。** [固定树](https://api.github.com/repos/ustclug/Linux101-docs/git/trees/4099db4a1f90e8e89cac3eca4bf2e36b2932e3f1?recursive=1)返回 `truncated: false`；逐项计数 `docs/` 下 35 个 `.md` 文件合计 484,174 字节，含答案、附录、命令及 Markdown。第一章中的时效性统计有引用，其他章节未逐篇核实。材料偏系统操作和教学，代码与命令密集；筛选自然段时要避免丢失关键前提，不能据此推断适合日常语言的大规模覆盖。

## 3. 《动手学深度学习》中文版第二版

**原站与实际文本。** [在线书站](https://zh.d2l.ai/)提供章节及作者、译者信息；[中文仓库说明](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/README.md)称新版先以英文编写再搬回中文版。固定提交 `e6b18ccea71451a55fcd861d7b96fddf2587b09a` 的[引言源码](https://raw.githubusercontent.com/d2l-ai/d2l-zh/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_introduction/index.md)返回 200，含较长现代中文段落。在线版可能变化，未逐段验证它与固定源码一致。该书提供经过组织的教学文本，但此处不把它归为纯原生中文。

**许可与权利链。** 该提交的[构建配置](https://raw.githubusercontent.com/d2l-ai/d2l-zh/e6b18ccea71451a55fcd861d7b96fddf2587b09a/config.ini)写明 `All authors. Licensed under CC-BY-SA-4.0 and MIT-0`，并记录英文来源仓库与翻译信息；同一提交的[根目录 LICENSE](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/LICENSE)是 Apache License 2.0。现有证据**不足以精确划定各许可对文字、程序、图片和外部数据的适用范围**，也不足以证明三者可任意择一使用；使用前须向维护者确认，并核查英文上游、译者贡献、引用和图表。不得只依据仓库的自动许可标签作判断。

**规模与限制。** [固定树](https://api.github.com/repos/d2l-ai/d2l-zh/git/trees/e6b18ccea71451a55fcd861d7b96fddf2587b09a?recursive=1)未截断；只统计 `chapter_` 开头目录下的 `.md`，并排除文件名以 `_origin.md` 结尾的文件，共 140 个、2,133,601 字节。它仍包含目录页、程序、公式和参考文献；排除原文副本不等于去除所有英文或重复。主题高度集中，涉及书中外部数据集的名称和示例，不意味着这些外部数据的权利随教材授出。

## 4. 文化机构：有具体资源与条款，正文取得受阻

文化部[无障碍资源清单](https://opendata.culture.tw/frontsite/barrierFree/openDataList)可直接读取，其中列有数个不同机构的中文资源。该站的[许可正文](https://opendata.culture.tw/cms/7650923)是**政府资料开放授权条款第 1 版**，不是 CC BY 4.0；条款称其与 CC BY 4.0 相容，并规定显名及对第三人著作人格权的尊重。只有具体资源确实依该条款释放且对应权利得到证实，才可将该条款用于对应内容。以下没有把目录描述、下载次数或字段数算成正文规模。

1. **国立台湾历史博物馆／国家文化记忆库：艺术与人文。** [资源详情](https://opendata.culture.tw/frontsite/barrierFree/openDataDetail/752)标示提供单位、中文、政府资料开放授权条款第 1 版、2025-12-26 更新及一个 JSON 接口；字段包括 `description`、`contentLicense`、`originalUrl`、建档单位与素材详情链接，表明单条描述和图像许可应分别核验。[实际接口](https://tcmbdata.culture.tw/opendata/dataSet/culture?subject=ART_AND_HUMANITY)本次返回 403，未读取任何具体正文，也无条数或长度证据。来源是多单位收存素材，资源页声明不能替代逐条来源及描述许可；同站[族群与语言类](https://opendata.culture.tw/frontsite/barrierFree/openDataDetail/756)同样列出逐条许可字段，不能视为独立已核验正文。可能涉及人物、口述、社群知识和敏感文化资料。
2. **文化资产局：传统工艺。** [资源详情](https://opendata.culture.tw/frontsite/barrierFree/openDataDetail/725)标示中文、提供机构及同一条款，并列 `historyDevelopment`、`registerReason`、`caseUrl` 等字段；资源页记载的文件更新时间为 2022-08-12，页面更新为 2026-09-07，二者不可混作正文版本。[JSON 接口](https://data.boch.gov.tw/opendata/v2/assetsCase/4.2.json)本次超时，未取得具体个案或总数。人物信息、地方传统知识、转载与登记文书需审查；目录承诺和字段名不足以证明各段连续、现代或适合训练。
3. **中正纪念堂管理处：建筑生态导览。** [资源详情](https://opendata.culture.tw/frontsite/barrierFree/openDataDetail/798)标示中文、同一条款和 `name`、`content`、`images`、`link` 字段，给出的资源更新时间为 2023-01-11，页面更新为 2026-07-07。[导览原站](https://tour.cksmh.gov.tw/)返回页面框架，但本次未读到正文；[对应 JSON](https://themedata.culture.tw/opendata/object/Point.json?origin=tour.cksmh.gov.tw&lang=zh-tw&branch=home)返回 403。尚无实际篇数、长度、固定内容或引用来源证据，图片权利亦不能由文本许可推定。

这些条目值得后续通过正常可访问渠道逐条核对；访问失败不代表永久不可用，也不能把目录数字当作已取得文本。

## 后续审查顺序

1. 先对《代数学方法》各卷和《Linux 101》按固定提交抽取少量独立段落，核对作者、贡献、第三方引用和许可，再统计可保留汉字及连续段落长度；避免将教材中的公式、示例或原文引述混入。
2. 对中文版深度学习教材，先确认三个许可声明的适用边界和英文到中文的权利链，再考虑抽样；对文化资源先解决正文访问，记录具体版本、原文地址、逐条许可与提供单位后再计算规模。
3. 所有候选进入任何训练流程前，检查事实时效、敏感资料、作品去重、已有语料与评测集重叠；分别报告发现、实际获取、通过权利审查及通过质量审查的数量。公开许可适用范围和发布义务须按实际用途与适用法律判断。

网络请求使用 `curl --ssl-no-revoke`：保留证书链校验，但未完成证书吊销检查。部分服务器返回 403 或超时；对没有读到的正文、未比对的版本及未取得的授权均不作肯定推断。
