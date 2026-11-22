# 孢子的多条件追问与完整回答

## 方法

基于上一轮使用孢子自身对话历史训练的检查点，继续更新第 24、27 层的普通对话残差。训练样本要求同一条回复同时给出已更正的信息和仍生效的条件。例如更新集合时间时保留原房间，或者改写句子时更换物品并保留人物与动作。

训练材料为 24 个自编三轮场景。每个场景都有人工整理的完整历史，其中 12 个还采集了冻结父模型自己生成的前两轮回复。旧的短答样本和项目已有常规对话参与回放。训练给完整追问分配五分之八的步骤，另三种训练样本分别覆盖前两轮、旧短答和常规回复。长答的监督位置分批投影到词表，每次最多四个位置，按词元位置交错分组，从而覆盖答案中段及结束位置。

评测使用另写的 12 个三轮场景。每个模型把自己的真实回复带入下一轮。末轮分别记录新信息、未变更条件、旧值回流、句数和是否自然结束，逐项都通过才记作整题通过。自动核对以文本模式为依据，同时对照原始回答检查语义。另复用原 24 道短答和混合日期记忆用例。训练和评测的用户问句做了逐字排除，小型对照确认 12 条参考答案、缺项及过时答案的判定，以及不同答案长度的分组覆盖。

这种多轮实生成和逐条约束检查分别对应现有的 MT-Eval、Multi-IF 与 FollowBench 的评测思路。本轮题目和分数属于项目内的本机实验。[MT-Eval](https://arxiv.org/abs/2401.16745) · [Multi-IF](https://arxiv.org/abs/2410.15553) · [FollowBench](https://arxiv.org/abs/2310.20410)

## 训练结果

| 配置 | 完整追问 | 原 24 道短答 |
| --- | ---: | ---: |
| 父检查点 | 0/12 | 11/24 |
| 完整回答训练 176 步 | 0/12 | 5/24 |

训练更新 98,304 个参数。最终追问、上下文前两轮、原短答和常规回复分别有 36、48、30、86 条训练请求，实际监督位置分别为 391、77、64、80 个。采集、训练和评测用时约 243 秒，峰值 CUDA 分配为 4.94 GiB。

训练后有一道题同时给出最新出库顺序和保留的贴签要求，但拆成两句，未通过题目的一句话要求。其它题仍会只说「玉米汁」而省去常温、无糖以及另一人的饮品。预约提醒还沿用了旧时间并加上提前十五分钟到达的要求。候选保留为实验检查点，父检查点仍是后续对照。

独立重开复现了 12 个场景的生成，五条日期读取及五条澄清与父检查点逐项相同。原检查点、父数据库及训练输入文件的摘要均通过复核。

核对逐题输出时发现时间关键词可能把「十一点半」算进「十一点」。重新判分严格区分分钟和数量，并核对包裹题要求的人名、数量、地点顺序。修订沿用全部原题和生成记录，两个检查点的完整追问总分均为 0/12。

## 对话历史对照

保持用户说过的事实和末轮要求相同，比较三种输入。第一种使用真实两轮助手回复。第二种保留轮次，清空前两条助手文本。第三种把三条用户消息按原顺序合成一条。以下字段得分是 12 道题中全部更新项和保留项的合计。旧值题数按出现明确旧信息的题计算。

| 模型 | 历史输入 | 更新项（共 15） | 保留项（共 23） | 出现旧值的题 | 完整通过 |
| --- | --- | ---: | ---: | ---: | ---: |
| 孢子父检查点 | 真实回复 | 4 | 3 | 1 | 0/12 |
| 孢子父检查点 | 清空助手文本 | 6 | 4 | 0 | 0/12 |
| 孢子父检查点 | 合并用户消息 | 7 | 9 | 2 | 0/12 |
| 孢子新候选 | 真实回复 | 5 | 7 | 1 | 0/12 |
| 孢子新候选 | 清空助手文本 | 5 | 8 | 1 | 0/12 |
| 孢子新候选 | 合并用户消息 | 5 | 12 | 4 | 0/12 |
| 千问 1.5B 指令版 | 真实回复 | 9 | 20 | 3 | 3/12 |
| 千问 1.5B 指令版 | 清空助手文本 | 10 | 18 | 3 | 4/12 |
| 千问 1.5B 指令版 | 合并用户消息 | 9 | 20 | 5 | 3/12 |

合并消息让较多的原条件出现在孢子回复里，同时抬高了旧值回流。新候选在出库题中，真实历史下给出了新顺序，合并后又改回先取蓝漆。清空助手文本也无法让孢子完整执行三轮指令。

千问在真实历史下的三个通过题经复核分别为双人饮品、恢复日期与未知原因、带贴签要求的出库顺序。清空历史时，按词匹配还会把「在阳光下感到暖」算作保留取暖动作，说明自动分数需要抽查。这组数据显示，助手文本和消息布局可以影响哪些条件出现在输出中。孢子对多条件的掌握还受语言骨干及训练方式的限制。

## 延迟结束对照

在新候选的真实历史上只改变末轮的结束限制，至少生成 12 个词元后才允许结束。中位生成长度从 14.5 增至 25，更新项从 5 增至 6，保留项从 7 增至 12，出现旧信息的题从 1 增至 3，完整通过仍为 0/12。骑行题补上了头盔，却把原来的向北改成向南。包裹题补齐三项，用三个句子代替要求的格式。增加长度可以让更多字段出现，却无法保证正确组合。

本轮保留原父检查点作为后续起点。下一阶段应验证解码器内部的逐轮状态更新，把应用新值和保留其它条件分别核对，再测试是否能联合生成完整回答。新版评分下独立重开仍逐条复现 12 个场景，相关的七项单元测试通过。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$parent = 'runs/spora-dialogue-correction-v1/student'
$out = 'runs/spora-dialogue-composition-v1'
.\.venv\Scripts\python.exe -u scripts/train_spora_dialogue_composition.py --checkpoint "$parent/bootstrap.pt" --database "$parent/individual.sqlite" --fixed runs/spora-dialogue-constraints-v1/result.json --previous runs/spora-dialogue-correction-v1/histories.json --output $out
.\.venv\Scripts\python.exe -u scripts/audit_spora_mixed_calendar_chat.py --checkpoint "$out/bootstrap.pt" --database "$out/individual.sqlite" --output "$out/calendar.json"
.\.venv\Scripts\python.exe -u scripts/train_spora_dialogue_composition.py --checkpoint "$out/bootstrap.pt" --database "$out/individual.sqlite" --verify "$out/result.json" --output "$out/reload.json"
.\.venv\Scripts\python.exe -u scripts/audit_spora_dialogue_composition_context.py --checkpoint "$parent/bootstrap.pt" --database "$parent/individual.sqlite" --record "$out/result.json" --phase before --output "$out/context-parent.json"
.\.venv\Scripts\python.exe -u scripts/audit_spora_dialogue_composition_context.py --checkpoint "$out/bootstrap.pt" --database "$out/individual.sqlite" --record "$out/result.json" --output "$out/context-candidate.json"
.\.venv\Scripts\python.exe -u scripts/audit_spora_dialogue_composition_context.py --qwen D:/SPORA-runtime/models/Qwen2.5-1.5B-Instruct-989aa798 --output "$out/context-qwen.json"
.\.venv\Scripts\python.exe -u scripts/audit_spora_dialogue_composition_context.py --checkpoint "$out/bootstrap.pt" --database "$out/individual.sqlite" --record "$out/result.json" --minimum-answer-tokens 12 --output "$out/context-minimum.json"
```
