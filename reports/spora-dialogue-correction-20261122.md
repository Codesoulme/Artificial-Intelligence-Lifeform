# 孢子在自身对话历史上的纠错训练

## 结构

在 DeepSeek 孢子日期记忆检查点的第 24、27 层增加两个秩 16 的残差。两处残差初始输出为零。解码请求携带神经记忆状态时，两处残差直接返回原有层输出。普通对话仍沿用相同的解码器，可以使用已经训练过的残差。这种做法将普通回复的参数调整限制在两个模块里，并利用已有的记忆读取状态控制激活。

训练器保存残差权重和激活模式。候选使用单独的个体数据库，沿用原个体的神经状态和日期记录，并绑定新检查点摘要。保存时逐个比较已有检查点张量和结构字段。

## 对照

30 组项目自编训练场景等量覆盖事实更正、较早限制和新限制。每组有两条背景消息和一条追问，追问题的答案从用户提供的事实推导。每组训练使用同一条追问和同一条目标回复。第一组使用人工写的两轮助手回复，第二组使用冻结孢子实际完成的两轮回复。两组从相同随机初始化开始，使用相同的样本次序、优化器、步数、监督输出词元预算和普通对话保持项。输入历史的长度随回复内容变化。

训练只采用本轮人工写好的目标回复。当前千问结果沿用作本机底座对照。训练历史与评测用户问句做了逐字排除。评测复用 24 条固定历史短答和 8 个三轮自行续聊场景，另用混合对话脚本重开候选个体并读取日期记忆。

## 各组结果

| 配置 | 更正 | 较早限制 | 最新限制 | 严格短答 | 自行续聊自然结束 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 日期记忆孢子检查点 | 0/8 | 0/8 | 0/8 | 0/24 | 18/24 |
| 人工整理历史 | 4/8 | 4/8 | 4/8 | 12/24 | 20/24 |
| 孢子生成历史 | 3/8 | 4/8 | 4/8 | 11/24 | 23/24 |
| 孢子生成历史，扩大常规回放 | 4/8 | 3/8 | 4/8 | 11/24 | 23/24 |

每组训练 128 步，更新 98,304 个参数。30 条短答监督包含 109 个答案词元。两组训练和评测共约 388 秒，峰值 CUDA 分配 4.94 GiB。孢子生成历史中的上一条回复有时保留了已撤销的值，例如用户改为浅紫，模型仍说「登记备用颜色为墨绿」。目标短答由较新的用户消息决定。

两种训练都提升了固定历史短答，人工整理历史多答对一题。孢子生成历史组在最新限制题没有逐字给出旧答案，人工整理历史组出现两次。开放追问中，孢子生成历史组能按下雨条件选择在家看电影，问到朋友需要确认什么时仍复述看电影。两组都能给出周四晚上八点，最终总结却遗漏影院门口。两组也都把纸张数量沿用为七张，漏做减法。

两组在独立进程中打开各自的新检查点和个体数据库。在混合对话的 21 轮里，各自的五条日期读取和五条澄清记录与父检查点逐项相同，包括旧版观察窗口、日期选取、矩阵读值和输出答案。日期数据库复制自原个体第 129 版状态，候选数据库拥有新的本地版本序号。训练过程中原检查点和原数据库的文件摘要一致。41 项相关测试通过，覆盖普通对话激活、记忆读取旁路、不同批次行的原有记忆掩码以及检查点和日期入口。

孢子生成历史组从独立进程重开后，两组对话评测的 48 条回复、词元数、结束标记和短答判定逐项复现。逐题训练结果、日期结果和重载结果分别为 `runs/spora-dialogue-correction-v1/student/` 下的 `result.json`、`calendar.json` 和 `reload.json`。

首轮普通对话保持池最终只有三条问候样本。针对这个不足，第二轮加入不同长度的解答、改写、边界询问和价值观等项目已有材料。保持步从四分之一增加到二分之一，并在这些步加入带标注回复的监督。第二轮使用同样的 128 步预算，差异包括回放覆盖、采样比例和损失组合。

第二轮首次用全部答案位置计算回放损失，在反向传播时出现显存分配错误。重试复用同一批已记录的模型历史。长回复保留完整前缀，词表输出只计算最多四个均匀选取的答案位置，包含首个词元和结束位置。小型 Qwen 对照核实了选中位置的分数与完整计算相同，也能反向传播。训练期间已用过的校验材料仍只用于评测。

## 扩大回放后的结果

扩大的回放池有 86 条普通对话请求。128 步中 64 步用于回放，共对 477 个采样到的答案位置应用标注监督。复用此前采集的历史后，训练和评测约 165 秒，峰值 CUDA 分配仍约 4.94 GiB。严格短答仍为 11/24，自然结束仍为 23/24。改约场景的最后一问从只复述时间退化为「场次」，禁蛋场景再次推荐鸡蛋清。对话可以自然停止，回答仍经常只复述历史里的一个词组。

扩大回放的检查点重开后也复现了五条日期查询和五条澄清。三组候选的个体状态摘要一致，两个训练记录所列的输入摘要在运行后通过复核。

三个候选保留为研究检查点。训练证明可用少量参数改变普通话语的约束跟随。下一轮的目标是让追问同时保留多个要素，例如更新时间时保留地点，并完整回答解释或改写请求。评测应分别检查新事实、未改变的条件和当前用户要求的回答形式。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -u scripts/train_spora_dialogue_corrections.py --checkpoint runs/spora-native-query-key-v1/bootstrap.pt --database runs/spora-calendar-sleep-v1/individual.sqlite --fixed runs/spora-dialogue-constraints-v1/result.json --steps 128 --output runs/spora-dialogue-correction-v1
.\.venv\Scripts\python.exe -u scripts/train_spora_dialogue_corrections.py --checkpoint runs/spora-native-query-key-v1/bootstrap.pt --database runs/spora-calendar-sleep-v1/individual.sqlite --fixed runs/spora-dialogue-constraints-v1/result.json --histories runs/spora-dialogue-correction-v1/histories.json --steps 128 --arm student --replay diverse --output runs/spora-dialogue-correction-v2-sparse
$candidate = 'runs/spora-dialogue-correction-v1/student'
.\.venv\Scripts\python.exe -u scripts/audit_spora_mixed_calendar_chat.py --checkpoint "$candidate/bootstrap.pt" --database "$candidate/individual.sqlite" --output "$candidate/calendar.json"
.\.venv\Scripts\python.exe -u scripts/audit_spora_chat_backbones.py --checkpoint "$candidate/bootstrap.pt" --database "$candidate/individual.sqlite" --fixed runs/spora-dialogue-constraints-v1/result.json --output "$candidate/reload.json"
$candidate = 'runs/spora-dialogue-correction-v2-sparse/student'
.\.venv\Scripts\python.exe -u scripts/audit_spora_mixed_calendar_chat.py --checkpoint "$candidate/bootstrap.pt" --database "$candidate/individual.sqlite" --output "$candidate/calendar.json"
```
