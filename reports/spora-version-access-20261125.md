# 把读取意图与询问版次分开判断

上轮已经验证从追问接上内部记录，但四次错误追问都给后续留下了读取机会。本轮比较四类动作头和独立读取意图头。只有确认原话要查记录时，才能直接读取或建立待澄清的会话状态。ClariEdge 把近邻负例作为区分真实歧义与表面歧义的主要手段；ReferClaw 则限制追问次数。孢子借鉴成对对照和一次追问两项思路。此次访问 ClariEdge 数据集接口未获授权，所有中文样本由本轮编写。 [上轮结果](spora-version-clarification-20261125.md) · [ClariEdge](https://arxiv.org/abs/2610.28686) · [ReferClaw](https://arxiv.org/abs/2611.07683)

## 方法

每组把同一条问句放进四种语境，分别要求读指定版本、澄清后读、以后再查和只评议表达。每组保持人物、事项和内层问句逐字相同。训练六组话术搭配四名虚构人物，共 96 条。另用两组新话术和不同人物校准，共 16 条。验收用三个新表达家族的 48 条，以及上一轮原来的 72 条。新题未参与校准，一个家族在四个事项上复用同一话术。

冻结 DeepSeek 前十二层、原版次头、时间轴和整值读出。重训同结构的四类动作头，并另练一个二分类小头。每个头训练 200 步。二分类首轮只判断用户是否请求查阅。整个成对组同时进入小批次，因此不能靠是否出现特定事项名来区分。阈值只用校准集，优先压低错误放行，再保留应通过的请求。实验复现旧动作头和两组新决策，并只让带独立校验的候选实际生成。

## 结果

| 首轮是否应进入读取流程 | 上轮动作头 | 成对训练 | 成对训练加独立校验 |
| --- | ---: | ---: | ---: |
| 新题应放行，24 条 | 22 | 24 | 24 |
| 新题错误放行，24 条 | 4 | 0 | 0 |
| 新题四种意图全对，12 组 | 6 | 12 | 12 |
| 上轮题应放行，48 条 | 48 | 48 | 33 |
| 上轮题错误放行，24 条 | 7 | 1 | 0 |

这里的放行包括建立追问状态，实际读取还要经过姓名、事项和版次判断。新题有 12 条本应追问，带独立校验的候选追问了 8 条。对这 8 条分别用三种简短回复确认版次，24 个分支全部从相应内部状态读出了正确的值。另一种问句使用「录下的内容」，事项边界规则未接受这种说法。直接读取新题正确 6/12，另外 2 条读错版本，4 条未找到事项。

在上轮 72 条材料中，误发追问及由其触发的模拟读取，从 4 次与 12 次降至 1 次与 3 次。加 0.90 的独立校验阈值后都降为零，同时直接读对的数量从 32/36 降至 17/36。最后一个错误追问的通过分约 0.71，部分正常直接读取的通过分低于 0.3。这个分布说明单纯降低共同阈值会重新放行已知的错误。

用存档决策补做事后对照。只校验要开启追问的情况，直接读继续使用原事项和版次约束。历史材料恢复 32/36 条正确直接读取，同时仍拦住错误追问引发的 3 次读取。这是根据本轮结果选出的候选，只用存档决定复算，未重新生成被恢复的直接回复。要用独立的普通对话和记忆混合场景重新验证，特别是看否定、转题及对话引用会不会被直接当成查阅。正式聊天入口尚未加载这些权重。

下一步冻结本轮权重和判断条件，采用跨会话混合回放。借鉴 MemConflict，将已经更正的记录、不同情境下可以同时成立的记录、以及相似但无关的事项放在同一条连续对话里。分别评分是否触发读取、选中了哪份内部状态和最终答复。清空可见历史后再重启验证，避免把顺着上一句回答当成持久记忆。 [MemConflict](https://arxiv.org/abs/2605.20926)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$timeline = 'runs/spora-version-timeline-v2/versions.pt'
$reader = 'runs/spora-version-reply-actions-v1/heads.pt'
$versions = 'runs/spora-native-version-families-v1/heads.pt'
$actions = 'runs/spora-version-clarification-v1/action.pt'
$archived = 'runs/spora-version-clarification-v1/result.json'
$output = 'runs/spora-version-access-v1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_version_access $timeline $reader $versions $actions $archived $output
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_version_access $timeline $reader $versions $actions $archived $output --verify
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_version_access $timeline $reader $versions $actions $archived $output --pending-only
```

首次运行与独立重载各约 114 秒，CUDA 张量峰值约 3.47 GiB。重载复现 120 条首轮问句的各组决策、后续分支及 77 条实际生成的词元。28 项相关单元测试通过。两个小头和阈值保存于 `heads.pt`，逐轮结果保存在 `result.json` 与 `reload.json`，事后对照保存在 `pending-only.json`。
