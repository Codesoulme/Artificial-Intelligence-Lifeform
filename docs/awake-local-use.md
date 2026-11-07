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
