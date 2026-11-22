# 1536 词元候选的中段信息保留

在 1536 词元检查点上比较相同权重和相同个体。八个局部层共用 256 词元预算，生成时每层保留最多 255 个历史位置。固定前缀保留最初 64 个词元。两种分散布局均保留最初 16 个，再给较早位置留 48 个固定名额。一种每隔 31 个位置留一个词元，另一种每隔 128 个位置留连续 4 个。随着已保留位置进入窗口，最近未保留位置填满剩余名额。全局基线由同一检查点临时卸下窗口配置获得。

12 道长文题覆盖集合点、金额和收件人三种事实。每条事实放进同一份填充材料的开头、288、544、928 四个位置。生成限 24 词元，回答包含目标值即计为找回。另用 8 条受控记忆题检查内部张量在长提示下的输出，题目提供地址及版本。两组输入约 1024 词元。

| 注意力布局 | 找回长文事实 | 前两处中段位置 | 受控记忆精确输出 | 1024 词元后的键值张量 |
|---|---:|---:|---:|---:|
| 全局 | 10/12 | 4/6 | 8/8 | 28.00 MiB |
| 固定前缀 | 5/12 | 0/6 | 7/8 | 21.99 MiB |
| 分散单点 | 6/12 | 0/6 | 7/8 | 21.99 MiB |
| 分散短块 | 6/12 | 0/6 | 8/8 | 21.99 MiB |

两种分散布局都多找回了一道近末尾的金额题，六道中段题仍未找回。分散短块恢复了一例受控记忆答案。四种布局均测得 1024 词元处分段与整段计算的首选词元一致。静态完整对照耗时约 430 秒，峰值 CUDA 分配约 4.94 GiB。逐题数据见 `runs/spora-distributed-anchors-v1/result.json`。分散位置的选择借鉴稀疏长距连接的思路。 [LongNet](https://arxiv.org/abs/2307.02486)

## 完整读入后按问题保留位置

读入时使用全局注意力。八个局部层各从末尾 16 个查询位置计算归一化注意力，跨头平均并用五词元均值平滑，从较早位置选出 48 个保留项。这 48 项、篇首 16 项和近期位置共用 256 词元预算。完整读入对照在相同预算下只保留固定篇首与近期位置。两种方法都在首次生成前完成裁剪，逐项检查同一组事实和受控记忆。 [SnapKV](https://arxiv.org/abs/2404.14469)

| 布局 | 找回长文事实 | 中段位置 | 受控记忆 | 1024 词元后的键值张量 | 1023 词元读入耗时 |
|---|---:|---:|---:|---:|---:|
| 全局 | 10/12 | 4/6 | 8/8 | 28.00 MiB | 3.92 秒 |
| 完整读入后固定裁剪 | 5/12 | 0/6 | 8/8 | 21.99 MiB | 3.94 秒 |
| 完整读入后按问题选取 | 10/12 | 4/6 | 8/8 | 21.99 MiB | 4.36 秒 |

按问题选取逐题找回全局模型找回的同十条事实。两条收件人中段题三种方法均未找回。两种裁剪的续算首选词元在一条 1024 词元探针上与全局一致，后续回答仍可能变化。此处仅统计目标值出现在回答中的情况，多条回答带重复或无关扩写。键值张量缩小约 21.5%，全模型峰值 CUDA 分配仍约 4.94 GiB。逐项输出见 `runs/spora-distributed-anchors-control-v1/result.json` 和 `runs/spora-distributed-anchors-snapshot-v1/result.json`。

## 带日期和干扰项的复测

新写四组登记内容。每组在约 276 和 560 个填充词元处分别插入一月版和二月版，半数题把二月版放在前面。约 892 处加入同字段的其他活动记录。每组分别问一月值和二月值。目标值出现且回答中未包含另一版本或干扰值才算通过，另统计只返回目标值的格式正确率。题面新写，填充文本沿用同一份混合语料。

| 布局 | 二月更新值 | 一月历史值 | 仅输出目标值 | 受控内部记忆 |
|---|---:|---:|---:|---:|
| 全局 | 4/4 | 2/4 | 0/8 | 8/8 |
| 完整读入后固定裁剪 | 0/4 | 0/4 | 0/8 | 8/8 |
| 完整读入后按问题选取 | 4/4 | 1/4 | 0/8 | 8/8 |

按问题选取在公演领票的一月题上错答二月值，完整注意力答对该题。展览和维修的一月题两种方法都错答二月值。这组对照耗时约 325 秒，峰值 CUDA 分配约 4.94 GiB，压缩后的键值张量仍为 21.99 MiB。逐题记录见 `runs/spora-distributed-anchors-revisions-v1/result.json`。按问题选取作为可重载的实验配置，历史问答仍保留全局对照基线。

隔离源码中通过 58 项检查点、事务、解码和局部注意力回归，覆盖从检查点恢复该布局、按问题选取位置和核对生成状态大小。先前并行运行 GPU 评测与回归时，有两项事务子进程遇到系统内存不足。GPU 阶段结束后串行复跑全部通过。

## 按键值头组分配保留位置

把 48 个按问题选择的名额分成两半。前 24 个仍按所有注意力头的平均分挑选，两个键值头组各用 12 个名额补足本组分数最高的位置。各层存储一份共享位置集合。用上一轮的四组登记记录和两组新题，对照平均分与分组名额。生成后统计各局部层对值和整条带日期记录的保留情况。选取阶段只看注意力分数。 [Ada-KV](https://arxiv.org/abs/2407.11550)

| 分配方式 | 二月更新值 | 一月历史值 | 仅输出目标值 | 受控内部记忆 | 1024 词元后的键值张量 |
|---|---:|---:|---:|---:|---:|
| 所有头平均 | 6/6 | 2/6 | 0/12 | 8/8 | 21.99 MiB |
| 平均分加分组名额 | 6/6 | 2/6 | 0/12 | 8/8 | 21.99 MiB |

两种分配方式的正误逐题一致。平均分方案在原八题上的生成也与上次逐词元一致。公演题的旧值「西塔大厅」完整出现在八层中的两层，两种压缩均未在任一局部层保留带日期的旧记录全句。展览和维修的旧记录分别在至少六层完整保留，两种压缩和全局注意力仍答成新值。逐项记录见 `runs/spora-snapshot-group-comparison-v1/result.json`。分组试验耗时约 267 秒，峰值 CUDA 分配约 4.94 GiB。

## 强制保留新旧记录的因果对照

只问六条历史值，在每个局部层的 48 个旧位置名额里先预留一月和二月记录的完整位置。剩余名额按原平均分选择，题目、解码和总预算沿用分组试验。逐词元核对两条完整记录都保留在八个局部层。这个干预使用评测标注定位记录，部署时的选择器仍需自行定位。

完整保留后，历史题由 2/6 升至 3/6，原本答对的两条都保留。公演题从二月的「北桥大厅」恢复到一月的「西塔大厅」。展览、维修和校对仍全部错答二月值。只有公演题在这一干预下恢复。两种保留情况下均没有历史题只输出目标值。独立的受控内部记忆仍为 8/8，键值张量仍为 21.99 MiB。实验约 140 秒，原始结果见 `runs/spora-snapshot-evidence-control-v1/result.json`。

先前的失分有两条来源。裁剪会把日期与值拆开，公演题可以通过保留完整记录恢复。证据齐全时，模型仍会选错日期，需要在版本选择处单独改善。按头组分配保留位置继续用于对照。若尝试降低裁剪损失，下一轮可把完整记录作为选择单位，或保留部分专门读取长文的头。 [DuoAttention](https://arxiv.org/abs/2410.10819)

24 项局部注意力和检查点测试通过，覆盖两种选择方式的完整读入、定长生成状态，以及原有检查点重载。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$checkpoint = 'runs/spora-native-context-1536-v1/bootstrap.pt'
.\.venv\Scripts\python.exe -u scripts/audit_spora_distributed_anchors.py --checkpoint $checkpoint --output runs/spora-distributed-anchors-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_distributed_anchors.py --checkpoint $checkpoint --full-prefill-control --output runs/spora-distributed-anchors-control-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_distributed_anchors.py --checkpoint $checkpoint --snapshot-only --output runs/spora-distributed-anchors-snapshot-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_distributed_anchors.py --checkpoint $checkpoint --revisions --output runs/spora-distributed-anchors-revisions-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_distributed_anchors.py --checkpoint $checkpoint --snapshot-group-comparison --output runs/spora-snapshot-group-comparison-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_distributed_anchors.py --checkpoint $checkpoint --snapshot-evidence-control --output runs/spora-snapshot-evidence-control-v1/result.json
```
