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

要体验将其余 26 层也换成孢子合并投影的候选，可以运行：

```powershell
.\spora_chat.cmd --migrate-backbone
```

加上 `--position-table` 可启用从底座位置系数初始化的参数表。为这一候选指定独立清醒状态文件，重启时使用同一组参数：

```powershell
.\spora_chat.cmd --migrate-backbone --position-table --awake-state runs/spora-awake-local/position-table-awake.pt
```

该入口加载个体后在内存中改造语言骨干，保留原来的分组注意力、旋转位置编码、归一化与 SwiGLU 计算。名称、明确记录、同人物追问、跨进程读回和睡眠核对均已实机验证。与现行双层改造版的公开题结果一致，自由聊天能力仍待独立评测。[改造对照](../reports/awake-language-replacement-audit.md)

连续的普通问答会在当前会话里带入最多四轮实际对话，达到位置上限时优先保留最近的轮次。写入记录或转入事实查询时清除这段对话。退出后，普通对话窗口随进程结束，已写入的个体记忆仍可读取。过长的单条消息会提示分段发送。

本机候选还支持把第 18 层换成学习型绝对位置注意力。使用独立的运行文件启动：

```powershell
.\spora_chat.cmd --migrate-backbone --absolute-attention runs/awake-absolute-attention-memory-v1/attention.pt --memory-file runs/spora-awake-local/absolute-memory.sqlite3 --awake-state runs/spora-awake-local/absolute-awake.pt --ordinary-intent-head runs/deepseek-awake-ordinary-intent.pt
```

该版本的位置上限为 256 词元。更正后可用「齐宁上一次记录里最爱喝什么？」查旧值。运行状态会绑定候选权重摘要，重启需加载同一候选。

第 18 层也可独立试用拟合后的两处 LayerNorm。以下启动项组合了合并投影和共享位置表，并使用独立个体文件：

```powershell
.\spora_chat.cmd --migrate-backbone --position-table --layer18-layernorm runs/awake-block-components-v1/components.pt --memory-file runs/spora-awake-local/layernorm-memory.sqlite3 --awake-state runs/spora-awake-local/layernorm-awake.pt
```

重启时使用同一份候选权重。此选项加载归一化参数，沿用预训练前馈计算。

完整第 18 层孢子块也可在独立个体上启动。该块采用学习型绝对位置、LayerNorm 和 GELU，位置上限为 256 词元。

```powershell
.\spora_chat.cmd --layer18-native runs/awake-native-block-v2/layer-18.pt --memory-file runs/spora-awake-local/native18-memory.sqlite3 --awake-state runs/spora-awake-local/native18-awake.pt --ordinary-intent-head runs/deepseek-awake-ordinary-intent.pt
```

重启时使用同一候选和个体文件。权重目录中的 `result.json` 提供文件摘要。问候、个体名称、写入、追问、查询更正前的值和睡眠核对已在本机验证。[整块对照](../reports/awake-language-replacement-audit.md)

保留 25% 蒸馏改动的实验候选也已通过本机重启测试，可另用一份清醒状态和记忆文件体验。

```powershell
.\spora_chat.cmd --layer18-native runs/awake-native-interpolation-v1/layer-18.pt --memory-file runs/spora-awake-local/native18-interpolation-memory.sqlite3 --awake-state runs/spora-awake-local/native18-interpolation-awake.pt --ordinary-intent-head runs/deepseek-awake-ordinary-intent.pt
```

也可以同时加载校正后的第 12 层与完整第 18 层。使用独立的个体文件，重启时传入相同的权重路径。

```powershell
.\spora_chat.cmd --layer12-native runs/awake-layer12-output-v1/layer-12.pt --layer18-native runs/awake-native-block-v2/layer-18.pt --memory-file runs/spora-awake-local/calibrated12-memory.sqlite3 --awake-state runs/spora-awake-local/calibrated12-awake.pt --ordinary-intent-head runs/deepseek-awake-ordinary-intent.pt
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
