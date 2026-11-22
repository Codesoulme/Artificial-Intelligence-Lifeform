# 孢子的逐轮条件状态与解码读出

## 结构

给孢子增加会话内的条件状态。每条用户消息由一个学习到的写入头处理。它可以写入日期、钟点、地点、物品、人物、数量和方向，也可以对每项选择不写。模型把选中的片段存成子词序列。新值覆盖同一位置，其它位置保留。助手的回复只作为常规对话历史输入解码器。写入头处理用户消息，因此助手在上一轮答错时，错误文本不会直接变成新状态。

最后一层解码器通过交叉注意力读取活跃条件，并把读取后的向量加回残差。另一路对状态中的子词提供可学习的生成分数增量。两路从零输出初始化。训练只更新新增模块，冻结 DeepSeek 和孢子的原有参数。已有日期记忆查询独立执行。

这是会话内对显式值的实验。目前覆盖三类有明确条件的任务，包括约见、包裹和集合路线。状态的验证涉及新值覆盖和原值保留。常规对话的生成仍由同一个孢子解码器处理。

生成时从状态中偏置候选子词，借鉴了序列模型将生成概率与复制概率结合的思路。孢子实验的状态来源是逐轮更新得到的有限条件表。[CopyNet](https://arxiv.org/abs/1603.06393)

## 写入结果

两种写入头都使用项目自编素材。保留组共 24 个三轮场景，词面和模板与各自训练素材分开。写入头从每轮原文选择起止位置，第三轮追问也走相同的写入路径。

| 写入头 | 初始状态完整 | 更正正确 | 原条件保留 | 第三轮无误写 | 最终状态完整 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 字符网络 | 0/24 | 3/24 | 10/44 | 24/24 | 0/24 |
| DeepSeek 前 12 层及字符网络 | 16/24 | 12/24 | 37/44 | 24/24 | 11/24 |

两组训练规模和轮数不同，当前比较反映两种方案的端到端表现。上下文写入头有时将被否定的旧人名写回，也会把「向东南」截成「向东」。最终状态完整的 11 组包括七组约见、一组包裹和三组路线。

## 首轮读出

冻结底座，用标注状态训练读出模块 192 步。在另行保留的 12 个受控场景中，固定每场景的用户消息和模型实际给出的前两轮回复。逐题切换空状态、学习写入的状态、标注的最新状态和标注的初始状态。四个条件下完全正确的题数依次为 0、1、1 和 1。给正确状态后仍只有一题全对，说明写入准确度只是瓶颈之一。旧状态出现同样的完整得分，现有结果也不足以证明答案由状态里的新旧值决定。

原 12 个自行续聊场景使用原来的前两轮回复，末轮接入模块后仍为 0/12。约见题正确写入周日与十一点，生成时却反复输出「请」，说明能从原文写出字段和能完整、自然地回答是两个需要独立检查的能力。本轮采集、训练和评测约 198 秒，峰值 CUDA 分配约 4.94 GiB。

## 同文换值训练

固定书面提示，分别提供标注的初始和最新状态。每对样本的目标答案跟随状态改变。半数样本只保留末轮追问，以测试读出能否从状态中恢复前文值。训练还在目标子词恰好存在于状态时，直接监督它对应的注意力权重。写入模块沿用已训练的上下文检查点，读出模块重新从零输出训练。

| 输入状态 | 受控场景中出现的正确字段（共 36 项） | 出现旧值的题 | 完整通过 |
| --- | ---: | ---: | ---: |
| 关闭工作状态 | 9 | 1 | 0/12 |
| 写入头预测的状态 | 15 | 1 | 0/12 |
| 人工标注的最新状态 | 17 | 0 | 0/12 |
| 人工标注的初始状态 | 13 | 4 | 0/12 |
| 只给末轮问题和预测状态 | 2 | 0 | 0/12 |
| 只给末轮问题和标注的最新状态 | 2 | 0 | 0/12 |

这里的字段得分只检查目标词面是否出现。保持对话文本不变，切换标注的旧状态和新状态时，有两道题的更正字段随之变化。例如包裹题的收件人从林念初切换为李怀青。输出仍会重复、追加无关字段，或在耗尽词元预算前未能结束。自然措辞对照仍为 0/12。三类受控任务使用有限字段和固定标注格式，结果不能外推到自由聊天或长期记忆。

换值训练及评测约 219 秒，峰值 CUDA 分配约 4.94 GiB。两轮各核对了五条日期读取和五条澄清，接入读出前后逐项一致。独立进程重载换值候选后，复现了 12 个自然场景的写入与回答，以及上述十条日期相关记录。四项单元测试覆盖字段覆盖与保留、写入输出形状、日期读出隔离和读出过程中断后的清理。下一轮先用人工标注的状态验证如何完整生成并正确结束，再合并写入头复测。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$parent = 'runs/spora-dialogue-correction-v1/student'
$before = 'runs/spora-dialogue-composition-v1/before.json'
$out = 'runs/spora-dialogue-working-state-v1'
.\.venv\Scripts\python.exe -u scripts/train_spora_dialogue_working_state.py --output runs/spora-dialogue-working-state-v1-writer
.\.venv\Scripts\python.exe -u scripts/audit_spora_dialogue_working_state.py --checkpoint "$parent/bootstrap.pt" --database "$parent/individual.sqlite" --before $before --output $out
$paired = 'runs/spora-dialogue-working-state-v2'
.\.venv\Scripts\python.exe -u scripts/audit_spora_dialogue_working_state.py --checkpoint "$parent/bootstrap.pt" --database "$parent/individual.sqlite" --before $before --writer-from "$out/working-state.pt" --contrast --output $paired
.\.venv\Scripts\python.exe -u scripts/audit_spora_dialogue_working_state.py --checkpoint "$parent/bootstrap.pt" --database "$parent/individual.sqlite" --before $before --adapter "$paired/working-state.pt" --verify "$paired/result.json" --output "$paired/reload.json"
```
