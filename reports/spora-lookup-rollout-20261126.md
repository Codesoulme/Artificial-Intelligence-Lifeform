# 连续改口后的记忆版次读回

沿用孢子双循环层候选、已有候选动作头和三版记忆档案。参考 StateWise，把查阅范围的变更与实际执行分开评分。基线复用现有动作头，另一路在其基础上用 72 条视图训练 120 步，继续预测每项的保留、选入和移出。训练和留出分别使用不同措辞和人名。 [StateWise](https://arxiv.org/abs/2610.33766) · [ToolTurns](https://arxiv.org/abs/2611.14925) · [单次更新基线](spora-lookup-transitions-20261126.md)

留出包含 18 段模拟对话，每段从已确认的一项开始，接连执行五次变更，每次变更后都追问版次。模型每次沿用自己的预测范围。恰有一项时用现有读出头复原档案值。候选地址、初始范围和版次由评测提供，状态更新只编码当前变更句。本轮没有走 Windows 聊天入口。三种路径共享部分措辞和地址，因此 90 轮中包含重复出现的变更句。

| 指标 | 原动作头 | 继续训练 |
| --- | ---: | ---: |
| 留出动作均正确 | 21/36 | 21/36 |
| 状态正确 | 60/90 | 48/90 |
| 应读轮次实际读对记录、版次和值 | 15/42 | 21/42 |
| 无需读取却读取 | 0/48 | 21/48 |
| 应为空时仍读取 | 0/24 | 12/24 |
| 五次变更状态均正确 | 0/18 | 0/18 |

继续训练后，模型将「我暂时不用调阅任何一条」判成两项均保留。它在三组地址和三种路径里共出现九次，每次都读取了本应清空的旧地址。随后以「接着按手头的安排」延续了三次错误读回。其余九次误读发生在原本应保留两项的轮次。这个结果表明单次动作正确数相同，持续使用时仍可能产生不同的误读。下一轮应单测全局撤销及其否定和转述，并用空、单项、双项三种起始状态验证后续裸版次请求。

本轮用时约 121 秒，权重和档案摘要一致，35 项相关测试通过。逐轮读回、状态和离线权重位于 `runs/spora-lookup-rollout-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_rollout runs/spora-lookup-transitions-clauses-v1 runs/spora-lookup-rollout-new --recurrent @weights
```
