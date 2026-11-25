# 分句证据与内部记忆读取

上次逐字段三分类仍会在「地点那一栏留空」的请求里读出地点。本轮检验给每条读取判断附上支持分句，能否减少这类错误。模型将候选人物及字段放在问题前，用冻结 DeepSeek 前 12 层提取每个分句末尾的因果表示。训练小头判断该分句是否请求、排除或未涉及这个字段。同一字段以最后一条非空判断为准。对照用同一输入、同一批 342 条自编请求和同样的 12 轮更新，只看句末表示及最终目标。另重载上一轮的二分类头。 [字段级对照](spora-field-authority-20261126.md) · [Schema-Guided Dialogue](https://arxiv.org/abs/1909.05855) · [Hy-MultiTurn](https://arxiv.org/abs/2607.29196) · [演化状态评测](https://arxiv.org/abs/2608.19652)

18 条留出请求使用训练未见的人名和措辞。11 条应读一项或两项，7 条应跳过，含中途撤销、指向未登记的人、引用和多人不明确等情况。逐条标注最终应读集合及支持分句。三种模型都对实际选中的记录执行张量读取并生成短答，使用同一份虚构人物档案和同一读出头。没有在这批题上调阈值。候选人物仍由目录中可见姓名确定，无法区分多个已登记人物时弃权。

| 指标 | 上一轮二分类 | 同数据整句 | 分句证据 |
| --- | ---: | ---: | ---: |
| 选全应读记录并生成全部正确值 | 1/11 | 3/11 | 2/11 |
| 无需读取却读出的请求 | 1/7 | 0/7 | 0/7 |
| 多读字段 | 6 | 0 | 1 |
| 漏读字段 | 6 | 10 | 10 |

分句方案执行的 4 次字段读取有 3 次引用了标注支持分句。一道要求补读许昀芝书名的改口题正确用到最后一句，并从档案读回「山谷晚风」。一道用黎书然作被排除对象的请求，原方案读回她的饮品，两个新头均未读取。另有一句「我不问何青玟在哪儿碰面，只提醒她把说好的东西捎来」，分句方案把第二句也当成读地点的证据，同时输出了地点和用品。两道需要两个字段的请求在两个新头上均未选全。分句偏好在措辞变化时仍会造成漏读，改善主要来自更少触发读取。七条显式版次回归全部通过，SQLite 档案和源权重摘要一致。49 项版次单元测试通过。首轮约 88 秒，CUDA 张量峰值约 3.47 GiB。

独立重载约 49 秒，复现全部 18 条自然请求和 7 条显式回归的路由、分句及生成结果。逐题记录位于 `runs/spora-clause-read-v1/`。

本轮两种模型保留在研究目录。整句分类、增加排除类别、分句监督都已验证。后续先改变数据和接入策略。使用人工对话中的真实请求检验模型能否把人物、字段和肯定或排除的动作绑在一起。同时沿用已核验的显式版次，将改造重点移到一条从请求到持久写入、重启读回、成品输出的完整路径。这能独立量化结构能力和开放中文能力。 [StaleBench](https://arxiv.org/abs/2611.19752)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaRun = 'runs/spora-clause-read-new'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_clause_read runs/spora-composition-training-v1 runs/spora-dual-version-reply-v1 runs/spora-field-authority-v1 $sporaRun
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_clause_read runs/spora-composition-training-v1 runs/spora-dual-version-reply-v1 runs/spora-field-authority-v1 $sporaRun --verify
```
