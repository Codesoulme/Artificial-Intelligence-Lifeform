# 本机版次记忆聊天

在 `D:\SPORA` 双击 `spora_version_chat.cmd`，或在 PowerShell 运行以下命令。

```powershell
cd D:\SPORA
.\spora_version_chat.cmd
```

入口加载本机的 DeepSeek-R1-Distill-Qwen-1.5B、双项读出头和口语版次头。项目目录应当已有 `.venv`、前两轮试验生成的权重和本地 DeepSeek 权重。启动会校验模型和试验文件摘要。首次打开使用独立的空档案，位置在 `runs\spora-version-chat-local\versions.sqlite3`。需要另一份档案时，可指定 `--profile`。 [双项读出文件生成方法](../reports/spora-dual-version-reply-20261125.md) · [口语版次文件生成方法](../reports/spora-version-normalizer-20261125.md)

```text
/记 林语 | 饮品 | 豆浆
/记 林语 | 饮品 | 红茶
请查林语的饮品当前版。
那上一版呢？
那首版呢？
```

用 `/记` 确认写入。同一姓名和字段再次登记会留下新版本。在命令末尾加 `| YYYY-MM-DD` 可指定生效日，默认用本机当天。同一天登记多次仍可查询首版、上一版与当前版。按日期查有效值时，同一天的多次写入以最后一条记录为准。

目前支持饮品、见面地点、用品和书目。登记值须能分成一至四个词元，适合短名称。重名时用不同的中文显示名。记录会先经过一次实际张量读回，成功才写盘。

普通消息交给本机语言模型，使用最近两轮聊天文本。查阅缺人物、事项或版次时会具体提示。转题时清除查阅范围，重新写明姓名和事项再查。`/清空对话` 清除本轮文本和地址范围，保留已确认的记录。`/帮助` 显示指令，`/退出` 结束会话。普通中文对话仍可能答偏，记忆读出当前主要适用于简短事实。

明确说「查」或「核对」后，可以分轮补充姓名、字段和版次。例如先说「请查谁的饮品上一版」，孢子会问姓名。回答一个已登记的姓名后，它按刚才的字段和版次读回。也可以先说「查林语的记录」，再回答「饮品当前版」。如果刚才报错了姓名，可以说「不是林语，是周小宁」，再补齐要查的事项。取消、清空对话、另起话题或退出后，需要重新发起查阅。 [澄清回放](../reports/spora-lookup-clarification-20261126.md)

刚查到一项时，也可以说「刚才核对过的那份记录，麻烦再查一下上一版」。如果只说「请再核对刚刚查到的那项」，孢子会先问要哪一版。刚查了两项或重启后，孢子会请你补姓名和事项。若已经说了「上一版」，回复「林语的饮品」即可。

同时查两项已登记的记录时，可以先说「请查林语的饮品首版；再查周小宁的书目」。孢子会追问周小宁的书目要哪一版，回复「当前版」即可完成两项查阅。如果两项都没给版次，按追问逐项作答，或回复「两项都查首版」。另起查阅、取消、清空对话或重启后，需要重新说明两项记录。

```powershell
.\spora_version_chat.cmd --profile runs\my-version-memory --reply-tokens 96
```

## 用两项记忆写一条消息

启用成句模型后，可以指定同一人物的两个字段、各自的版次以及已经确认的时间。取出后，由本机模型按写作要求生成句子。已有档案可直接打开。 [成句模型准备](../reports/spora-typed-composition-20261125.md)

```powershell
.\spora_version_chat.cmd --profile runs\my-version-memory --composition runs\spora-typed-composition-v1
```

```text
/记 许林秋 | 见面地点 | 杏叶廊
/记 许林秋 | 用品 | 青布袋
/记 许林秋 | 用品 | 灰毛帽
/成句 许林秋 | 见面地点当前版 | 用品上一版 | 周五傍晚 | 给许林秋写一句能直接发的微信，说明周五傍晚在哪里碰面、需要带什么。只给消息正文。
```

可组合见面地点与用品、见面地点与书目、书目与饮品。版次可以选当前版、首版、上一版或指定第几版。时间按输入原文使用，支持四个词元以内的短表达。写作要求决定句子要表达的动作。可以先用普通查阅核对所需版次。 [本机写入及重启对照](../reports/spora-local-composition-20261126.md)

启用同一个成句模型后，还可以在查阅同一个人的两项记录后，直接引用刚才的结果。

```text
请查许林秋的见面地点当前版；查许林秋的用品上一版。
按这两项写一句消息，说明在哪儿碰面，还要提醒对方带上要带的东西。
周五傍晚。
```

没给出时间时，孢子会先追问。也可以在写作要求末尾加一句「时间定在周五傍晚」。此方式沿用刚选中的版次。转题、取消、清空或重启后需要重新查阅两项。 [多轮回放](../reports/spora-dialogue-composition-cli-20261126.md)

## 试用循环注意力版

准备好孢子父检查点和第 19、22 层权重后，可以在同一个本机入口登记、查阅和成句。

```powershell
$sporaParent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaLayer19 = 'runs/spora-gelu24-recurrent-layer19-v2'
$sporaLayer22 = 'runs/spora-gelu24-recurrent-layer22-v1'
.\spora_version_chat.cmd --profile runs\my-version-memory --composition runs\spora-typed-composition-v1 --recurrent $sporaParent $sporaLayer19 $sporaLayer22
```

循环版支持原有版次档案，启动会检验父检查点、循环层权重和词汇档案的绑定。每次模型前向最多处理 2048 个词元位置。遇到长度提示可用 `/清空对话` 再试，已登记的记忆会保留。 [循环版回放](../reports/spora-recurrent-version-chat-20261126.md)

### 口语查阅试验

有本机训练产物 `runs/spora-spoken-query-data-v2/public.pt` 及同目录的 `result.json` 时，可以在循环版加上 `--spoken-lookup`。

```powershell
.\.\spora_version_chat.cmd --profile runs\my-version-memory --recurrent $sporaParent $sporaLayer19 $sporaLayer22 --spoken-lookup runs\spora-spoken-query-data-v2
```

模型在识别到适合查阅的口语请求后，会先说出猜测的人物和事项。确认时可回复「是，上一版」。也可先回答「是」，再补充所需版次。若猜错，可以说「不是林语的饮品，是周小宁的饮品首版」。换话题、取消、清空或重启后，请重新说明要查的内容。遇到模型未提出查阅的问法，可以直接给出姓名、事项和版次。 [跨轮回放](../reports/spora-spoken-lookup-dialogue-20261126.md)

### 在线状态

加上 `--awake-state` 可让清醒模块在进程空闲时继续运行。默认加载 `runs/p24-awake/core-seed7.pt`，用 CPU 做小步更新。用同一份状态文件重启时，需要保持同一档案和模型权重。

```powershell
.\spora_version_chat.cmd --profile runs\my-version-memory --recurrent $sporaParent $sporaLayer19 $sporaLayer22 --awake-state runs\my-version-memory\awake.pt
```

`/status` 显示运行状态，`/focus` 提高更新频率，`/relax` 降低频率，`/sleep` 暂停更新，`/wake` 恢复。单独查到一项记录后，短时间内可以说「再说一遍刚才查到的那项」。入口从在线张量取回那次选中的版次，再从档案读出名称。复述只能使用一次。等待过期、取消、转题、清空、睡眠或重启后，可以重新指定姓名、字段和版次。 [在线回放](../reports/spora-recurrent-awake-chat-20261126.md)

## 本机回放

回放从空测试档案开始，使用 Windows 启动入口发送登记和查阅消息，然后退出并在独立进程重开。脚本只使用位于指定结果目录下的新档案。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.audit_spora_version_local_chat runs\spora-version-local-chat-audit-new
```

本机测试登记六份张量记录。首轮 27 项及重启后 8 项回放全部通过，分别用时约 94 秒和 85 秒，含模型加载。49 项版本相关单元测试通过。同一天给同一饮品字段连续登记三次，当前、上一版和首版分别读出「绿豆汤」「枸杞茶」「果茶」。转题、取消、翻译和涉及缺失对象时，检查了实际查阅记录编号。退出重开后，省略姓名的第一句要求澄清。补上姓名后读回原张量，连续追问也能查到两个旧值。重启前后档案摘要一致，完整性检查为 `ok`。

首次回放中，翻译「那上一版呢」受上一轮地点回答干扰。单独切开历史后，同句仍错译，整理书桌答复变成泛化拒答。最终保留普通文本承接，只清除过期的查阅范围。取消查询给出简短确认。普通生成仍出现忽略「只说一个办法」和回复截断，回放通过数仅统计入口、写入、查阅及澄清的行为。 [多轮对话退化研究](https://arxiv.org/abs/2505.06120) · [记忆状态追踪研究](https://arxiv.org/abs/2608.19652)
