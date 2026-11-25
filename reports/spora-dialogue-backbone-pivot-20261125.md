# 中文多轮更正的底座对照

用 DeepSeek 自动整理现行要求，12 道开发题均未通过。改为学习设置、保留和取消条件后，留出对话的最终状态仍为 0/6。人工用自然中文写清最新要求，当前孢子也只答对 1/6。下一轮暂停扩训练集和继续调这个字段头，先复核回答能力的上限。 [工作状态实验](spora-dialogue-revisions-20261125.md)

已有权重迁移实验显示，把千问的部分权重混入 DeepSeek 未改善中文完整对话，部分条件还使记忆回答退化。完整迁移 Qwen2.5-1.5B-Instruct 到孢子结构，曾保留普通对话表现。用已知地址和版次校准记忆回答层后，受控读取恢复到 32/32。本轮用后来暴露问题的更正任务重新核对该方向。 [权重迁移实验](named-memory-weight-transfer-20261124.md)

## 对照设计

复用 12 条三轮开发对话和 6 条措辞不同的留出对话。两种千问配置都按实际答复进入下一轮，再用同一模型直接回答人工合并的最新要求。使用原生千问格式、贪心解码、48 词元回复上限和 256 词元总位置预算。DeepSeek 与当前孢子的成绩来自此前同题研究，沿用各自使用的 DeepSeek 直接回答格式。对比包含各底座相应的聊天格式。

原版和接入孢子结构的千问使用同一份固定版本权重。迁入版本加载原有位置表初始化、记忆接口及已校准的回答层，本轮只评估普通生成，所有调用均使用进程内临时对话。脚本核验底座文件、检查点、校准产物及重建的个体状态摘要。逐题报告位于 `runs/spora-dialogue-backbone-pivot-v1/result.json`。

## 结果

| 生成模型 | 开发题分轮 | 开发题现行要求 | 留出题分轮 | 留出题现行要求 |
|---|---:|---:|---:|---:|
| 原版 DeepSeek-R1-Distill-Qwen-1.5B | 0/12 | 5/12 | — | — |
| 当前 DeepSeek 孢子 | 0/12 | 4/12 | 0/6 | 1/6 |
| 原版 Qwen2.5-1.5B-Instruct | 6/12 | 6/12 | 3/6 | 5/6 |
| 接入孢子结构的千问 | 5/12 | 6/12 | 3/6 | 5/6 |

千问两种配置的 36 个最终回复都正常结束，其中 33 个逐字一致。接入后开发集少通过的题是解释露水，输出含「水汽」且给出正确过程，但漏掉了明确要求的「小水滴」。两版都在候车提醒中沿用了已取消的雨衣，并在茶会清单中保留了已取消的香橼。给出人工整理的现行安排后，两题都通过。评分只覆盖可以从文本检查的要求。例如有的结果虽然通过，仍自行加上了用户未要求的邀请措辞。

这组题中，完整指令底座带来的提升大部分保留到了孢子结构。对留出题，结构对照的 3/6 和 5/6 为后续设定了两个起点。先用千问孢子研究如何保持现行要求，同时保留 DeepSeek 孢子作对照。接入版是否可用于常规交流，仍需独立的开放中文对话、自然记忆写入和重启后的联合回归。本轮未训练或切换本机入口。

整轮 GPU 对照约 305 秒，CUDA 张量峰值约 2.96 GiB。三项对话状态回归通过。 [DeepSeek 与当前孢子的开发题结果](spora-dialogue-evidence-20261125.md) · [留出题结果](spora-dialogue-revisions-20261125.md)

## 何时改变方向

先指定一项外部可观察指标，例如留出场景完整通过数、明确取消的执行率，或重启后的新旧值读回正确率。如果两次迭代在同一项上没有改善，暂停增加训练规模。同题比较原版底座、接入结构的候选和给定正确状态的生成。原版与改造版同时出错时先看语言模型或任务数据。给定状态大幅改善时才把训练投入状态更新或寻址。改造版单独退化时用成对样本定位产生差异的层及解码接口。

11 月的 AFVE-Bench 区分采纳新要求、泄漏已废弃内容和答复能否直接使用。后续中文复核可以分开记录这三个结果。微软的 Lost in Conversation 有开源的成对任务构造与运行代码，作者在英语上验证，适合复用生成对照的思路。中文长交互的 Hy-MultiTurn 则提供转题干扰、对象定位和约束整合等更接近下一阶段验收的任务分类。 [AFVE-Bench](https://arxiv.org/abs/2611.10541) · [Lost in Conversation 代码](https://github.com/microsoft/lost_in_conversation) · [Hy-MultiTurn](https://arxiv.org/abs/2607.29196)

Qwen3.5-0.8B 是可以独立比较的新底座。文档和固定配置显示它默认直接回答，24 个文本块以三层线性注意力加一层完整注意力的方式交替，约 8.73 亿总参数。结构上与当前 Qwen2 加载器不同。若要比较，先用独立环境跑相同的中文原样题，测显存、速度与结果，再决定是否值得做结构迁移。本轮只核对了模型资料。 [官方权重及说明](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$checkpoint = 'runs/named-memory-layer9-packed-reply-v1/packed.pt'
$donor = 'D:\SPORA-runtime\models\Qwen2.5-1.5B-Instruct-989aa798'
$calibration = 'runs/named-memory-qwen-structure-control-v2/result.json'
$result = 'runs/spora-dialogue-backbone-pivot-v1/result.json'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_dialogue_backbone_pivot $checkpoint $donor $calibration $result
```
