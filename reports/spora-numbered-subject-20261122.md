# 编号历史查询的姓名补全与跨页复验

在已经明确「第几次更正前或后」的问句中，尝试从更正短语前后的语法边界提取完整姓名，再与已有登记指纹核对。实际查询沿用原有读取意图判断。只有旧定位器漏选且提取到唯一已登记姓名时补回地址，明确姓名未登记、提取到两个人或两个定位结果冲突时拒绝读取。姓名从短语中整段核对，相似长名的短前缀无法代替全名。

重建两个隔离状态，把同一人物的第三次更正放到下一页。在前一页中先写入初值和前两次更正，并用无关地址填满剩余槽位。分别在原有句式和新写的问句上，对比「原定位器加编号路由」与「加上姓名补全」。新负例覆盖前置和后置的未登记长名、双主体、编号前后同时询问、超出历史的编号和翻译引用。每个正确结果仍要求两个状态均选对事件并生成不同的正确值。

## 结果

| 问句 | 路由 | 编号问句选对事件 | 两份状态均答对 | 应拒绝却读取 |
|---|---|---:|---:|---:|
| 原有句式 | 原定位器 | 29/48 | 28/48 | 0/32 |
| 原有句式 | 加入姓名补全 | 48/48 | 47/48 | 0/32 |
| 新写句式 | 原定位器 | 33/48 | 32/48 | 5/48 |
| 新写句式 | 加入姓名补全 | 48/48 | 46/48 | 0/48 |

三条剩余的编号失败都已选对内部事件和记录值，其中一份状态的解码回答却只给出数字。两套题也各有八条无编号的当前值查询。新写的八条都未找到姓名，当前值定位需要单独改善。每个状态有 72 条观察与两页。独立进程重载两份状态后复现全部逐题结果，状态摘要一致。初跑和复验分别用时约 168 秒和 195 秒。证据在 `runs/spora-numbered-subject-v1/`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$arguments = @(
  '--checkpoint', 'runs/spora-native-query-key-v1/bootstrap.pt',
  '--route-source', 'runs/qwen-compatible-gelu-layer17-v1/integrated.pt',
  '--router', 'runs/qwen-compatible-gelu-lexical-boundary-v1/hybrid',
  '--temporal', 'runs/spora-past-override-v1',
  '--output', 'runs/spora-numbered-subject-v1',
  '--extended'
)
.\.venv\Scripts\python.exe scripts/audit_spora_numbered_revisions.py @arguments
.\.venv\Scripts\python.exe scripts/audit_spora_numbered_revisions.py @arguments --reopen
```
