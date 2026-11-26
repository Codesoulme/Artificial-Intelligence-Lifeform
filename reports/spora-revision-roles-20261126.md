# 孢子版本记忆中的字段角色与读取顺序

成句模型在去掉旧对话后能读出两项正确值，却经常把地点、负责人或材料放进错误语法位置。本轮把角色随记录直接送入读出头。每条记忆仍由已有低秩张量重建，保留版次和寻址逻辑。类型描述如「地点」「人物」，角色进一步区分展示地点与归还地点、录音者与剪辑者。相同类型的两项使用已有 `MemoryAddress.condition` 区分，独立保存和选版。 [前一轮成句对照](spora-revision-composition-20261126.md) · [Xing USP](https://arxiv.org/abs/2610.34343)

冻结第 19、22 层循环恢复版、词表和个体。四组共用 48 条成句训练样本，每条按两种顺序提供记录，共 96 条训练视图，每组更新 240 次。反转记录顺序时，教师句保持不变，只交换复制动作的目标位置。三组用同样大小的字段嵌入和输出头，分别输入固定占位标签、基础类型或细分角色。第四组使用对每条记录共享的打分函数并对记录表示平均汇聚，已读标记跟随记录，词表动作不依赖记录排列。第四组的参数结构与前三组不同。 [Deep Sets](https://arxiv.org/abs/1703.06114) · [Pointer Networks](https://arxiv.org/abs/1506.03134)

留出包括八道已有的不同类型题，以及四道新增的同类型题。新题涉及两个地点或两位人员，训练与评估的完整值及问句分开，训练包含同类话题。每题评估当前和旧版，两个版本分别倒置记录顺序。可见输入仅含末问，状态由实验指定。严格通过要求两项值正确、没有竞争版次、值和动作匹配、正常结束。只写对两个值单独计分。

| 读出结构 | 不同类型，正序 / 反序分别用对 | 同类型，正序 / 反序分别用对 | 同类型，两种排列均用对 | 同类型换序逐词一致 |
| --- | ---: | ---: | ---: | ---: |
| 只用位置 | 0/16 · 4/16 | 0/8 · 6/8 | 0/8 | 0/8 |
| 基础类型 | 4/16 · 4/16 | 4/8 · 2/8 | 0/8 | 0/8 |
| 细分角色 | 4/16 · 4/16 | 6/8 · 6/8 | 6/8 | 7/8 |
| 细分角色，共享打分 | 5/16 · 5/16 | 6/8 · 6/8 | 6/8 | 8/8 |

仅提供基础类型时，两个地点仍得到同一类型标签。把摆放和归还分成不同角色后，六条当前或旧版的同类型回答在两种记录顺序下都用对了两项值。共享打分组把换序逐词一致的数量提高到 24/24，并让两种顺序下的 48 条答复均含正确两值且正常结束。严格角色通过在不同类型题上仍为 5/16。分数严格和词面判定各自有局限，样本由少数话题构成。四组普通持久读取均为 32/32。源权重、个体和原版本数据库一致。81 项相关测试通过，其中新增用例验证同一次交换同时移动角色与已读标记时，复制分数随之交换，文字分数保持一致。

共享组重新训练约 97 秒，权重摘要及逐条生成与首次训练一致。独立进程按摘要加载权重和数据库，再换用六条新的问法，沿用相同事实。交换顺序后仍有 24/24 条逐词一致，同类型规则通过仍为 6/8，不同类型由 5/16 降到 2/16。所有答复的两个值及版次仍正确。自动规则也有漏判，例如把「请把周五午后录音时间，由温向晓剪片头」判断为角色匹配，但这句话仍需要重写。下一轮应在保留共享读取的前提下改进成句监督与人工复核，尤其检查动词与所引用值之间的关系，以及同一校验规则对改写的泛化表现。

## 复现

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$parent='runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$reference='runs/spora-revision-version-state-v1/result.json'
$layout='runs/spora-hybrid-position-recovery-v1/19-22'
$layers=@("$layout/layer-19", "$layout/layer-22")
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_roles $parent $reference runs/spora-revision-roles-v1 @layers --mode positional --mode type --mode role
$shared='runs/spora-revision-roles-shared-v2'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_roles $parent $reference $shared @layers --mode shared
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_roles $parent $reference runs/spora-revision-roles-shared-fresh-v1 @layers --mode shared --replay $shared --fresh-questions
```
