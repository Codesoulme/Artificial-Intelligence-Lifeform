# 长对话中从版本化神经状态读出两项事实

此前将登记和更正移近能找回单项新值，但同一句话要求两项时仍常漏答。本轮固定约 1,400 词元的对话和末问，交换同一地址的当前、旧版及空状态。沿用孢子的快权重写入法，以原始会话消息的词元及解码层表示建立低秩张量。保存后关闭并重新打开实验版本库，由原版次选择器选取每一项地址对应的生效版本，再从张量重建值词元。词元直接进入模型读出，三个条件给模型的消息及问题文字完全相同。StaleBench 与 StateMemBench 都将读到哪一版与最终答复分开评估。 [长对话因素对照](spora-revision-factors-20261126.md) · [StaleBench](https://arxiv.org/abs/2611.19752) · [StateMemBench](https://arxiv.org/abs/2608.19652)

测试覆盖地点与联系人、录音时间与剪辑员、标签材料与存放点、数量与接收人四个场景，各使用两种取值。第五个场景的两个字段都映射到同一会话槽位，第二种取值还有两个五词元短语，超过现有版本编解码器的四词元上限，因此两题未纳入比较。地址、写入跨度和应读版次由实验提供。从重新打开的版本库读回的 32 项值均与预期字段和原文一致，实际保存了 24 条独立版本。先验正确的地址和选版使这个实验能测出从内部读出到回答的损失。

骨干为相同的第 19、22 层循环恢复版。前两组复用同一个会话读出权重，第一组按词元合并语言分布和槽位分布，第二组额外使用值的连续跨度和已输出字段。第三组复用早期双字段读出头，将完整的字段作为词元序列动作。两个模型读出头都来自此前训练，在本轮的场景和提示上没有更新。 [会话读出训练](spora-memory-conditioned-reply-20261124.md) · [整段读取动作](spora-dual-version-reply-20261125.md)

## 同提示下交换当前与旧版

| 读出方式 | 当前版两项完整 | 旧版两项完整 | 两版均完整 | 空状态含最新两项 |
| --- | ---: | ---: | ---: | ---: |
| 逐词元混合 | 0/8 | 0/8 | 0/8 | 0/8 |
| 逐词元混合，带跨度连续性与覆盖 | 1/8 | 0/8 | 0/8 | 0/8 |
| 整段动作，允许随时结束 | 0/8 | 0/8 | 0/8 | 0/8 |
| 整段动作，有未覆盖字段时屏蔽结束符 | 3/8 | 2/8 | 2/8 | 0/8 |
| 整段动作，过早结束时先读出待答字段 | 8/8 | 8/8 | 8/8 | 0/8 |

完整通过要求两项均出现，竞争版次的值未出现，并生成结束符。逐词元覆盖组有 5/8 道题随状态改变文字，出现过混合姓名和地点词元、只答人名或改变任务意图等问题。整段动作屏蔽结束符时，在同一问句的两种状态下可以分别输出「杏仁石亭。林越河。」和「松街校门。林越河。」。另有题持续生成标点，直到剩余长度不足以复制待输出的完整值。允许直接结束时，当前和旧版各有四条答复只复制了首项，第二项均未完整出现。三轮评测分别约 222、138 和 103 秒。空状态的文字在各读出方式间一致，与同模型的原长题逐条相同。32 条既有具名读取均正常结束、内容精确。

补全动作保留原有的词元和字段竞争。只有模型选了结束符、已确认字段还有待答项且预算允许时，才按当前动作分数读出一个待答字段，再继续生成。新的因果对照中，16 条带状态的答复均输出正确的两个跨度并结束，耗时约 106 秒。独立进程约 101 秒，逐词复现了这 16 条答复及 8 条空状态答复。49 项版本存储、时间选择、循环注意力、覆盖及聊天持久化测试通过。实际答复仍多为「二十一份。薛临泉」一类短片段，缺少用户要求的交接语义。同一类型多个字段、超过四词元的值、自动寻址和自然成句需要分别验证。

去掉可见前文，只保留末问和已选中的内部版本，又复测了 24 条生成。当前、旧版各 8/8 条均含对应两项并结束，8 条空状态没有出现任何保存值，与普通生成一致。16 条带状态答复中有 6 条与带前文时逐词相同，其余仍输出对应版本的完整事实，标点及措辞出现变化。这项受控结果确认字段来自重新加载的版本张量。该轮约 60 秒，地址和版次仍由试验明确提供。

接下来的生成训练可保持版本状态及确认逻辑，给动作头同时监督片段出现的位置和字段间自然措辞，并在未见过的题型中按用途评判整句。动作头训练时使用的是另一批短请求，长对话中的失败反映了从内部状态走到答复时的分布差异。近期研究同样观察到，内部可读出状态时，最终行为仍可能遗漏约束。独立的自动写入实验还需验证更正和原记录能正确归到同一地址。 [Lost with a Map](https://arxiv.org/abs/2609.33883) · [TransMem](https://arxiv.org/abs/2611.10683)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$parent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$layout = 'runs/spora-hybrid-position-recovery-v1/19-22'
$reader = 'runs/spora-memory-conditioned-reply-v1'
$run = 'runs/spora-revision-version-state-v1'
$layers = @("$layout/layer-19", "$layout/layer-22")
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_version_state $parent $reader $run @layers
$span = @('--span-reader', 'runs/spora-dual-version-reply-v1', '--reference', "$run/result.json")
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_version_state $parent $reader runs/spora-revision-version-span-v1 @layers @span
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_version_state $parent $reader runs/spora-revision-version-span-early-stop-v1 @layers @span --allow-early-stop
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_version_state $parent $reader runs/spora-revision-version-span-complete-v1 @layers @span --complete-on-eos
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_version_state $parent $reader runs/spora-revision-version-span-complete-reload-v1 @layers @span --complete-on-eos
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_version_state $parent $reader runs/spora-revision-version-span-question-only-v1 @layers @span --complete-on-eos --question-only
```
