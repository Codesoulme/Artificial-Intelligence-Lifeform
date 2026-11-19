# 本机持续活动研究入口

在D:\SPORA双击awake.cmd。它创建独立会话副本，运行在CPU，只有窗口运行期间才有持续更新；未安装开机服务。默认chat.cmd仍是原有中文入口，本版本尚不理解任意中文目标。

输入demo：提供一次从节点0到10的规划目标，之后不再输入，观察内部循环产生下一步建议。focus提高频率与规划预算；relax降低频率；sleep暂停；wake唤醒后需要新环境观察，可再次输入demo；status查看实际计数；quit保存退出。建议不会自动操控电脑。

每次启动显示本次SQLite状态文件。继续同一份工作状态可运行：

```powershell
.\.venv\Scripts\python.exe -m spora.awake_cli --resume-session runs/你的会话.sqlite
```

新进程保留目标、参数及工作记录，但使旧传感状态失效，需重新提供观察。任意JSON观察格式为type=observe、graph为16×16二值邻接矩阵、position为0..15、source为user/sensor/synthetic、urgent为布尔。目标格式为`{"type":"goal","target":10}`。source是调用方声明，不是来源真实性识别器。

短演示：

```powershell
.\.venv\Scripts\python.exe -m spora.awake_cli --demo-seconds 5
```

此阶段的输入是有限数字环境，持续运行和已学规划不等于中文理解、人类意识或完整做梦。原个体与语言检查点保留。[设计和验证](P24-awake-design.md)。

## 开始与 DeepSeek 版孢子聊天

在当前电脑双击 `D:\SPORA\spora_chat.cmd`，或在 PowerShell 执行：

```powershell
cd D:/SPORA
.\spora_chat.cmd
```

启动后输入「你叫什么名字？」。若要记录事实，可输入 `/记住 齐宁最喜欢喝米香玄米茶。`，再问「齐宁平常最爱喝什么？」和「那他的职业是什么？」。当前入口支持饮品、职业和到访地点，示例使用已登记的测试人物。按 `Ctrl+Z` 再按回车可正常退出。

记忆保存在 `runs/spora-awake-local/memory.sqlite3`，首次记录时创建。在线状态文件 `awake.pt` 与名称张量及记忆路径绑定。`/status` 查看状态，`/relax` 放松，`/focus` 专注，`/sleep` 暂停更新，`/wake` 恢复。此入口允许最多三轮经过核验的连续改问。

进入睡眠时核对最近 16 次写入中的最新事实。若结果报告差异，可使用 `--verify-replay 16` 单独核对，或用 `--repair-replay 16` 从记录的快照恢复。修复事务会检查同页的其他已登记事实，并把修复与证据版本写入同一个提交。

在另一份本地检出中，如果已具备当前分支的底座与实验检查点，可以按下列步骤重建缺少的三个本地文件。脚本会拒绝覆盖已有输出。

```powershell
New-Item -ItemType Directory -Force runs/spora-awake-local
.venv/Scripts/python.exe scripts/deepseek_awake_recall_intent.py --output runs/spora-awake-local/recall-eval.json --head runs/spora-awake-local/recall-intent.pt
.venv/Scripts/python.exe scripts/deepseek_awake_subject_switch.py --output runs/spora-awake-local/subject-eval.json --head runs/spora-awake-local/subject-intent.pt
.venv/Scripts/python.exe scripts/deepseek_awake_self_name.py --output runs/spora-awake-local/name-eval.json --identity runs/spora-awake-local/self-name.pt --name '孢子'
```

实机通过名称读取、连续改问、更正和新进程重载。自由聊天和自然指代仍在评测。详细样本与限制见 [DeepSeek 在线状态实验](../reports/deepseek-awake-chat.md)。
