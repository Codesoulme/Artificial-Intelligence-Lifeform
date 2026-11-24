# 孢子循环注意力的长历史消融与对齐

对之前保存的十组更正值与两种历史长度做逐题对照。可见历史、位置表、生成参数和个体状态固定。父模型和原双层组沿用已保存的结果，读取时确认题目、输入摘要和个体摘要一致。四个新组在同一进程分别运行。 [长对话基线](spora-recurrent-dialogue-cost-20261124.md)

| 替换方式 | 约 700 词元完整正确 | 约 1,400 词元完整正确 | 约 1,400 词元命中新值 | 命中保留值 | 提到旧值 |
|---|---:|---:|---:|---:|---:|
| 父模型 | 4/10 | 4/10 | 4/10 | 6/10 | 3/10 |
| 两层原循环注意力 | 7/10 | 2/10 | 3/10 | 5/10 | 2/10 |
| 只替换第 19 层 | 5/10 | 1/10 | 3/10 | 4/10 | 2/10 |
| 只替换第 22 层 | 5/10 | 3/10 | 3/10 | 5/10 | 2/10 |
| 两层关闭远距摘要读取 | 5/10 | 2/10 | 2/10 | 7/10 | 4/10 |
| 两层的分块从 8 扩到 32 词元，复用原权重 | 5/10 | 1/10 | 1/10 | 4/10 | 3/10 |

在这批长题上，第 19 层单独替换就会退步。关闭两层的摘要后，保留值出现得更常，新值却出现得更少，回答中提到旧值的次数翻倍。扩大精确窗口会改变权重看到的局部范围和摘要内容，本轮仅测直接复用权重。读出的得分来自同一批十组值，完整通过还要求正常结束并排除旧值。其他未替换层始终有完整对话历史。

## 长样本对齐

从已保存的第 19、22 层循环权重分别起步。使用既有会话状态训练分区和本机整理的中文维基训练语料，核对语料散列。48 条训练序列各有 70–1,589 词元，两层各更新 128 步。训练问答与最终评测分开。把首次登记与更正放在不同位置，并混入 0、5、11 或 14 段无关材料。用父模型相应层的精确注意力向量监督，损失对生成答案的区域给更高权重。每层独立取父模型的原输入。只更新循环注意力的小型投影和混合门，保存两层后组合测生成。LoLCATs 的层内输出拟合和 MOHAWK 的逐步迁移为这类对照提供参照。 [LoLCATs](https://arxiv.org/abs/2410.10254) · [MOHAWK](https://arxiv.org/abs/2408.10189)

| 层号 | 约 1,400 词元留出题全序列误差 | 留出题末尾 16 个位置误差 |
|---|---:|---:|
| 19 | 0.523 → 0.301 | 0.592 → 0.310 |
| 22 | 0.485 → 0.274 | 0.602 → 0.382 |

误差为相对均方误差，每条留出序列分别归一化，再求平均。拟合阶段使用带预设答案的训练输入，留出误差在生成前计算。实际问答另做贪心生成。

| 双层权重 | 短更正 | 约 700 词元 | 约 1,400 词元 | 开放题规则通过组数 |
|---|---:|---:|---:|---:|
| 训练前 | 8/10 | 7/10 | 2/10 | 2/6 |
| 长历史对齐后 | 8/10 | 6/10 | 2/10 | 2/6 |

通过状态只变了一道。约 700 词元时，原循环版正确记得湖灯队和岗位，续训版答成旧队名白砂队。约 1,400 词元时，仍是两个志愿者场景同时写对新组名和岗位。对齐后的 32 条受控持久记忆答复与父模型逐项一致。局部拟合更接近教师，但本轮证据不足以采用新权重。后续训练需要直接衡量输出能否随状态更正而改变、能否完整生成新值，并在相同题上控制事实出现位置。

独立进程重载两份新权重，逐题复现 30 条更正、18 轮开放对话及 32 条持久记忆回答。个体状态摘要一致。5 项循环注意力单元测试通过。逐题消融结果在 `runs/spora-recurrent-history-ablation-v1.json`，对齐结果和两份权重在 `runs/spora-gelu24-recurrent-long-alignment-v1/`。

```powershell
$sporaBase = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaFirst = 'runs/spora-gelu24-recurrent-layer19-v2'
$sporaSecond = 'runs/spora-gelu24-recurrent-layer22-v1'
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_history_ablation $sporaBase runs/spora-recurrent-dialogue-extended-v1.json runs/spora-recurrent-history-ablation-v2.json $sporaFirst $sporaSecond
.\.venv\Scripts\python.exe -m scripts.train_spora_recurrent_long_alignment $sporaBase runs/spora-gelu24-recurrent-long-alignment-v2 $sporaFirst $sporaSecond
.\.venv\Scripts\python.exe -m scripts.train_spora_recurrent_long_alignment $sporaBase runs/spora-gelu24-recurrent-long-alignment-v2 $sporaFirst $sporaSecond --verify
```
