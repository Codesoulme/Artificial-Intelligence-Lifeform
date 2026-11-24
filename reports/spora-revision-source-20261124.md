# 确认操作中的值来源

从同一 DeepSeek 孢子检查点的前 20 层取当前轮分句表示，让分类头区分句内给出的值、明确引用待定值和无需取值。训练用 3,456 条合成文本，半数附带先前提议。用现有操作头和范围提取头回放六组共 31 轮。评测提供人物和字段。此前没有可靠提取到值的确认可回退到待定值。本轮另外执行来源判定，只有选中待定值来源时才允许确认继承旧提议。[范围提取实验](spora-revision-value-20261124.md)

| 值来源 | 选对 |
|---|---:|
| 无需取值 | 10/11 |
| 当前轮给出具体值 | 9/15 |
| 明确引用待定值 | 3/5 |

| 条件 | 当前状态正确 | 待定状态正确 | 完整正确会话 |
|---|---:|---:|---:|
| 无独立来源约束的旧回放 | 15/31 | 16/31 | 0/6 |
| 评测给定来源，模型选操作与值 | 15/31 | 16/31 | 0/6 |
| 模型同时预测来源、操作和值 | 12/31 | 12/31 | 0/6 |
| 评测给定操作和值，模型预测来源 | 16/31 | 20/31 | 0/6 |

独立来源能表达失败提取和确认继承之间的区别。当前学习模块的预测降低了逐轮状态正确数，评测给定来源也无法修复其他提取错误。实验在内存中的模拟记录执行，数据量和句式有限。该候选停留在离线评测。已验证的本地确认入口仍可使用明确命令。[本地待确认事实](spora-pending-structured-memory-20261124.md)

首次复核进程遇到 Windows 虚拟内存不足和 CUDA 驱动退出。复跑在加载并验证检查点后释放未被特征提取使用的后段解码层及词表投影，独立重载逐轮复现。权重和记录位于 `runs/spora-revision-source-v1/`。

```powershell
$sporaBase = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaActions = 'runs/spora-decision-transition-v2-focused/head.pt'
$sporaValues = 'runs/spora-revision-value-v2/value.pt'
.\.venv\Scripts\python.exe -m scripts.audit_spora_revision_source $sporaBase $sporaActions $sporaValues runs/spora-revision-source-v2
.\.venv\Scripts\python.exe -m scripts.audit_spora_revision_source $sporaBase $sporaActions $sporaValues runs/spora-revision-source-v2-reload --verify runs/spora-revision-source-v2
```
