# 孢子历史记忆的双时间查询

历史读取新增可选的 `known_by`。`as_of` 表示想查哪个生效时点，`known_by` 表示最晚可以使用哪一步收到的观察。后者要求完整的事件索引，只扫描截止观察步内的事件。按现在所知复盘时，已确认的迟到事实可以修正过去。按当时所知查询时，迟于截止步的观察不参与选择。现有人工确认数据缺少确认时点，历史视图按观察当时的待确认状态处理。

## 实机对照

在经过位置参数迁移的 DeepSeek 孢子检查点上，为两名模拟人物各构造两份五步历史。每份包含初次登记、更正、迟到的直接观察和两条其他人物的记录。迟到观察在第 4 步收到，但从第 1 步开始有效。测试为每个观察步映射一个日历日期，调用者显式提供人物地址、生效步和观察截止步。生成提示只包含人物、日期和请求。值由模型从选定的神经历史读出，使用现有输出适配器生成。

| 条件 | 正确路由 | 生成值正确或正确拒绝 |
|---|---:|---:|
| 补录到达前查询旧日 | 4/4 | 4/4 |
| 确认补录后按现在所知复盘同一天 | 4/4 | 4/4 |
| 尚未收到更正 | 4/4 | 4/4 |
| 收到更正当天 | 4/4 | 4/4 |
| 补录已见、当时尚待确认 | 4/4 | 4/4 |
| 查询日期早于补录生效步 | 4/4 | 4/4 |

20 条指定了观察截止步的路由全部与相应时刻实际保存的快照一致。两个世界里同样的问句会随内部记忆返回不同值。例如小林的 1 月 2 日记录，按补录前收到的信息回答「咖啡」或「果汁」，允许使用确认过的补录则回答「果汁」或「可乐」。20 条生成与 4 条受控拒绝均符合独立期望。首轮约 83 秒，峰值 CUDA 分配约 4.94 GiB。逐题记录保存在本机 `runs/spora-bitemporal-memory-v1/result.json`。

当前验证覆盖调用方给定时间和人物的路径。日历映射是试验配置，默认聊天入口仍需接入可持久的时钟和自然语言时间寻址。此处用同一份持续更新的神经历史查询旧事件，更长历史中后续写入是否干扰旧值或来源仍需单独验证。

独立进程从磁盘重载后，24 条路由、回答及快照对照逐项复现，耗时约 82 秒。46 项索引、来源、历史读取和检查点测试通过。

## 中文日期问句入口

新增只读入口。保存的个体同时携带观察日期和已登记姓名的指纹。日期坐标按日解析，截止日包含当天最后一条观察。在已覆盖的日期范围内，空白日按此前最后一条观察定位。入口验证保存的日期数与观察步一致，并核对个体与解码器的地址基底。回答仍由选中的神经历史及原输出适配器生成。

本次支持有限的中文饮品查询句式、ISO 日期和阿拉伯数字的「年／月／日」。试验保留从固定日期表导入的映射。日期范围以保存的记录为依据。完整的日历持续写入、相对日期和开放问题仍待接入。

两种问法共 48 条，包含 40 条正确生成及 8 条待核实记录的正确拒答。额外 20 条无效日期、超出日历、近似姓名、翻译请求和双日期请求也全部正确拒绝，且没有进入神经历史读取。首轮有 24 条问句因问号后附带「只回答饮品名称」而未被解析。补充该边界后复测全部通过，耗时约 85 秒。逐题结果见 `runs/spora-calendar-history-v2/result.json`。

Windows 管道首轮受系统默认编码影响。明确使用 UTF-8 后，独立控制台进程正确回答「咖啡」「果汁」并拒绝未登记姓名，退出后保存文件的摘要一致。50 项日期、版本读取和检查点回归通过。

## 持续写入和恢复

中文日期入口可以从便携个体建立独立的 SQLite 存储。每笔显式观察沿用神经写入控制器，默认附上本机当日日期。观察、日期索引和修订号在同一事务提交，姓名和日期用于定位，答案仍由模型读取神经矩阵。已有同一天的多条记录按当天最后一条观察确定查询截止步。指定较早生效日时，要求该日已有观察坐标。待确认的历史事实继续沿用现有确认机制。

实机在新事务个体追加「小林，牛奶」。第 5 步以更正写入，修订号变为 1，当日问句从历史矩阵读到了「牛奶」。源个体和解码器检查点的文件摘要一致。过期写入者被拒绝。分别在写首个张量后和事务提交后强制结束进程，前者恢复为空日历及零步观察，后者恢复为同一次新增观察及对应日期。53 项相关回归通过。

独立进程打开同一数据库后，从观察步 5、0 和 4 分别回答「牛奶」「咖啡」「果汁」，对应当日新增、按旧有资料查询、按补录复盘旧日。实际修订和日期清单位于 `runs/spora-live-calendar-v1/individual.sqlite`。两个进程的问答记录保存为同目录的 `live-write.jsonl` 和 `reopened.jsonl`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$common = @('--checkpoint', 'runs/spora-native-query-key-v1/bootstrap.pt', '--output', 'runs/spora-bitemporal-memory-v1')
.\.venv\Scripts\python.exe -u scripts/audit_spora_bitemporal_memory.py @common
.\.venv\Scripts\python.exe -u scripts/audit_spora_bitemporal_memory.py @common --reopen

.\.venv\Scripts\python.exe -u scripts/audit_spora_calendar_history.py --checkpoint $common[1] --prior $common[3] --output runs/spora-calendar-history-v2
.\.venv\Scripts\python.exe -u scripts/spora_calendar_history_chat.py --checkpoint $common[1] --individual runs/spora-calendar-history-v2/person-0-world-0.pt

$db = 'runs/spora-live-calendar-example/individual.sqlite'
$today = Get-Date -Format yyyy-MM-dd
.\.venv\Scripts\python.exe -u scripts/spora_calendar_history_chat.py --checkpoint $common[1] --individual runs/spora-calendar-history-v2/person-0-world-0.pt --create-database $db --observe 小林 牛奶
.\.venv\Scripts\python.exe -u scripts/spora_calendar_history_chat.py --checkpoint $common[1] --database $db --question "小林在$($today)的饮品是什么？"
```

在示例个体里，这两条问句可以查同一个生效日。

- 只按截至2026-01-04已收到的记录，2026-01-02对小林有效的饮品是什么？
- 帮我查小林在2026年1月2日当天登记的饮料是哪种？
