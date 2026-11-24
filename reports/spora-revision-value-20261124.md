# 孢子从更正原句提取新值

使用冻结 DeepSeek 孢子的前 12 层表示，从地点、数量、联系人和时间更正中选择一个连续字符范围或暂缓。字段由评测给出。训练包括 1,152 条样本，包含原值与新值同句、询问、待定以及只引用已有候选的确认。两个变体让候选范围的起止位置额外读取双向上下文，第三组分别优化范围排序和是否提取。训练和留出姓名、措辞和取值分开。原模型和持久个体没有写入。[操作分类实验](spora-decision-transitions-20261124.md)

| 条件 | 单值提取 | 原值与新值同句 | 无需取值时暂缓 | 原值与新值同句中正确范围排第一 |
|---|---:|---:|---:|---:|
| 原读法，零门槛 | 1/32 | 5/96 | 80/96 | 52/96 |
| 双向上下文，零门槛 | 5/32 | 15/96 | 76/96 | 63/96 |
| 分开训练排序与暂缓，零门槛 | 4/32 | 15/96 | 72/96 | 65/96 |
| 分开训练排序与暂缓，校准门槛 | 0/32 | 1/96 | 90/96 | 65/96 |

所有条件在应使用已有候选或仅查询的 32 条短题上均未提取新值。范围排第一是离线诊断，未绕开门槛执行写入。分开训练组在 128 条不同姓名和句式的校准题上以最高负例分差设门槛。留出题仍有 6 条误提取，同时大量合法值被挡住。

## 多轮状态回放

对 31 轮数据，评测仍提供对象和关系。实验分别使用给定操作和值、预测值、同时预测值和操作。没有选出值时，确认操作可以使用相同关系的待定值，因此回放仍未单独识别「明确要求采用待定值」和「这次没有可靠抽出新值」。该区分与交易语义一同决定可否进入正式写入。

| 值提取器 | 仅预测值时当前状态正确 | 同时预测操作和值时当前状态正确 | 完整正确会话 |
|---|---:|---:|---:|
| 给定操作和值 | 31/31 | 31/31 | 6/6 |
| 原读法 | 16/31 | 15/31 | 0/6 |
| 双向上下文 | 16/31 | 15/31 | 0/6 |
| 排序与暂缓分别训练，校准门槛 | 12/31 | 12/31 | 0/6 |

双向上下文改善了短句范围判断，但三组都未恢复完整会话。当前模型只在离线实验中使用。独立重载复现三组逐轮结果，18 项既有相关测试通过。模型和个体摘要一致。权重和结果保存在 `runs/spora-revision-value-v1/`、`runs/spora-revision-value-v2/`、`runs/spora-revision-value-v3/`。

```powershell
$sporaModel = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaActions = 'runs/spora-decision-transition-v2-focused/head.pt'
.\.venv\Scripts\python.exe -m scripts.audit_spora_revision_value $sporaModel $sporaActions runs/spora-value-reproduce --contextual --separate-loss --diagnostic
.\.venv\Scripts\python.exe -m scripts.audit_spora_revision_value $sporaModel $sporaActions runs/spora-value-reproduce-reload --verify runs/spora-value-reproduce --contextual --separate-loss --diagnostic
```
