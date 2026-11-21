# 多次更正的编号历史读取

使用已验证的标准检查点和姓名、时间分类器。新路径识别明确的「第几次更正前或后」短语，在该人物的事件键索引中读取神经来源信息，按观察顺序数已提交的更正。第一次登记、重复或支持记录和暂缓事件均不计为更正。读「之前」选该次更正前最后一次提交的值，读「之后」选该次提交时写入的值。答案仍由模型从对应神经记忆生成。

为八个模拟人物分别写入初值、三次更正和一条重复登记，跨人物交错写入。两份状态中，同一个问句指向同一次更正，但记录值不同。比较沿用最近版本的旧路径与编号路径。共 56 条正例，其中 48 条明确指定更正序号，另有 32 条测试不存在的序号、多个序号、缺少前后和引用翻译。正例计分要求两份状态都选对事件和生成值。

事件键索引保存每个事件属于哪个地址，值和更正来源仍来自神经张量。编号路径依赖完整索引，本轮从两个新的隔离个体开始写入并在读取前存盘重载。语法处理涵盖明确编号与前后关系，实际生效日期和多条件比较需要另做评测。

## 结果

| 项目 | 最近版本路由 | 编号历史路由 |
|---|---:|---:|
| 56 条正例，双状态均正确 | 9/56 | 29/56 |
| 48 条指定更正序号的正例，双状态事件正确 | 8/48 | 29/48 |
| 48 条指定更正序号的正例，事件与答案均正确 | 8/48 | 28/48 |
| 32 条负例中触发读取 | 20/32 | 0/32 |

给定正确人物地址后，编号路径在两份状态下均选对全部 48 条历史事件。自动端到端有 19 条编号问句未定位到正确姓名。第二次修订前和第三次更新前两种句式中，16 条均在姓名环节失败。另有一题事件和值都读对，生成却把「果汁」答成「3」。每份状态各用一页，写入 40 个事件，包含 32 次值提交和 8 条重复确认。

新进程重载两份保存的历史状态，逐项复现 88 条问句的选址、事件和回答。两次串行评测约用时 165 秒和 149 秒。28 项历史路由、时间分类和标准检查点相关测试通过。逐题证据在 `runs/spora-numbered-revisions-v2/`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$arguments = @(
  '--checkpoint', 'runs/spora-native-query-key-v1/bootstrap.pt',
  '--route-source', 'runs/qwen-compatible-gelu-layer17-v1/integrated.pt',
  '--router', 'runs/qwen-compatible-gelu-lexical-boundary-v1/hybrid',
  '--temporal', 'runs/spora-past-override-v1',
  '--output', 'runs/spora-numbered-revisions-v2'
)
.\.venv\Scripts\python.exe scripts/audit_spora_numbered_revisions.py @arguments
.\.venv\Scripts\python.exe scripts/audit_spora_numbered_revisions.py @arguments --reopen
```
