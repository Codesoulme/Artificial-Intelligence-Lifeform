# 更正事件评测停滞诊断

日期：2026-11-25。

## 评测中断与恢复

更正事件实验保存分类头 `decisions.pt` 后退出，缺少最终结果。第一次恢复期间，Windows 记录该 Python 进程占用约 6.74 GB 虚拟内存，系统剩余虚拟内存曾降至约 0.77 GiB，另一个 PowerShell 进程抛出 `OutOfMemoryException`。原始退出没有保留异常栈。评测达到内存峰值的具体调用仍未测定。

增加 `--resume-heads`，直接载入保存的权重。将输出层直接流式加载到模型设备，省去先在 CPU 暂存再移到 GPU 的步骤。第二次恢复取得写入和状态回放计数，随后因脚本缺少 `ExitStack` 导入退出。补齐导入后，完整评测约 184 秒，独立重载约 177 秒。两次状态回放与持久记忆答复逐条一致，受控持久读取为 32/32。Windows 在第二次恢复期间仍报告过虚拟内存压力，后续运行应继续监测。

## 多轮更正对照

冻结底座和跨度写入器。对同样的 24 组事实各构造 6 个后续决定，共 48 条应写入与 96 条应保留。直接组用一次全对话判定，其门槛取独立校准集的最高负例。结构组逐轮预测提议、确认、撤回或无关，由草稿状态决定是否提交。结构组用最大分数分类，两个运行点的拒写门槛不同。状态回放从给定的正确初值开始。

| 留出指标 | 直接判定 | 逐轮事件与草稿 |
| --- | ---: | ---: |
| 应写入时获准提交 | 7/48 | 15/48 |
| 应保留时误写 | 8/96 | 12/96 |
| 目标字段及完整新值均正确提交 | 1/48 | 1/48 |
| 完整状态正确（含应保留题） | 90/144 | 87/144 |

两组的误写全部出现在「先提出备选，随后切换无关话题」。结构组把 24 条无关回合中的 12 条当成确认。即使把写入决定替换为正确标签，现有候选在 48 条应写入样本中也只有 6 条同时找对字段和完整新值。这个上限来自复用同一组候选的反事实统计。继续调门槛无法修复候选绑定，当前试验保持在研究分支。

同场早期对照中，原版 DeepSeek 与日期记忆孢子检查点在 24 道严格短答题上均为 0/24，千问 1.5B 指令版为 12/24。另一组实验中，经过整层结构替换的版本出现额外语言回退。三字段问答直接提供正确状态时，两类问题也各只有 8/24 完整通过。这三组分别测量本机底座短答、结构替换影响和状态到答复的转换，适用范围以各自实验为准。 [底座短答](spora-chat-backbones-20261122.md) · [整层替换](awake-original-language-control-20261121.md) · [三字段问答](spora-observed-episodes-20261124.md)

## 下一步

优先比较在字段描述下复制新值的实验与现有按字段分类的路径。先给正确字段测完整值，再恢复自动定位。回合决策加入同一句「我同意」分别指向当前草稿和另一事项的对照。验收同时测精确写入、误写、长间隔重启读回和自然中文答复。字段描述与复制机制可参考 [Description-Driven Task-Oriented Dialog Modeling](https://arxiv.org/abs/2201.08904) 和 [TripPy](https://aclanthology.org/2020.sigdial-1.4/)，长间隔与更正评测可参考 [LongMemEval](https://arxiv.org/abs/2410.10813)。

结构迁移以 DeepSeek 原语言层为冻结对照，逐次比较每次局部改动对普通对话和记忆任务的影响。可用本机千问指令版做同题对照。恒等初始化与逐块迁移可参考 [LLaMA Pro](https://arxiv.org/abs/2401.02415)。

复现命令如下。

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$split = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
$sporaArgs = @(
    'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
    'runs/spora-revision-writers-v2',
    'runs/spora-revision-events-v1',
    "$split/layer-19", "$split/layer-22"
)
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_events @sporaArgs
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_events @sporaArgs --verify
```

已有 `decisions.pt` 且结果缺失时，用 `--resume-heads` 替代首次训练命令。
