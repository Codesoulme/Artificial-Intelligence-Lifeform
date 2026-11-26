# 角色问法增广与训练步数对照

此前共享角色指针头在原有不同角色题上为 5/16，在独立新问法上为 2/16，同类型题为 6/8。持久张量读取始终为 32/32，因此本轮只调整角色成句头的训练问法和更新步数。骨干、词表、角色编码、版本库和记忆状态均冻结。 [共享角色头](spora-revision-roles-20261126.md) · [新问法回放](spora-revision-roles-shared-fresh-v1)

在原有 48 条角色成句样本上增加六个场景的中文角色关系改写。改写强调两个值在句子中的语义作用，例如“先说牌子放哪儿，再说谁负责收回”。分别训练 240 步和 120 步。两组都使用共享角色指针头，并在独立进程中重载权重后评测。评测同时包含原有题、未见问法和两个同类型场景。值的读回、正常结束以及反转输入顺序均单独检查。

| 条件 | 原有不同角色题 | 独立新问法 | 同类型题 | 两种输入顺序逐词一致 |
| --- | ---: | ---: | ---: | ---: |
| 原共享头 | 5/16 | 2/16 | 6/8 | 24/24 |
| 增广，240 步 | 4/16 | 12/16 | 6/8 | 32/32 |
| 增广，120 步 | 11/16 | 11/16 | 6/8 | 32/32 |

两个增广版本在 16 条不同角色题中都正常结束并输出两个值。240 步版本对独立问法最好，但原有题回归明显。120 步版本在原有题和新问法之间更平衡，且保留同类型题成绩。它仍有 5/16 的原有题失败，不能作为正式聊天入口的默认成句头。空状态回复保持普通生成，持久记忆不受影响。

本轮结果说明表述增广有效，但更新过多会损伤原有角色关系。下一轮以 120 步候选继续做真实成句人工复核，重点修正“值出现但动作关系错误”的样本，再决定是否加入更细的动词关系监督。尚未将候选接入本机入口。相关训练脚本回归通过，产物位于 `runs/spora-revision-roles-augmented-120-v1` 和 `runs/spora-revision-roles-augmented-120-fresh-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$parent='runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$reference='runs/spora-revision-version-state-v1/result.json'
$layout='runs/spora-hybrid-position-recovery-v1/19-22'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_roles $parent $reference runs/spora-revision-roles-augmented-120-new "$layout/layer-19" "$layout/layer-22" --mode shared --augment-questions --steps 120
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_roles $parent $reference runs/spora-revision-roles-augmented-120-fresh-new "$layout/layer-19" "$layout/layer-22" --mode shared --replay runs/spora-revision-roles-augmented-120-new --fresh-questions
```
