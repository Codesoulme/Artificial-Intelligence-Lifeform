# 本机普通作答的解码和末层适配对照

已接入的版次聊天入口在查阅和重启回放中通过 35 项检查，普通对话仍会漏译、忽略简短作答要求。此前在另一个检查点上，限定原生推理用尽预算后只补结束标记。本轮直接复现本机入口的三条问题，检验结束标记后增加换行的效果，再测试小规模末层适配。 [本机入口](../docs/version-chat-local-use.md) · [此前推理预算实验](spora-open-reasoning-20261125.md) · [DeepSeek 用法与采样建议](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B#usage-recommendations)

## 解码对照

冻结本机 DeepSeek-R1-Distill-Qwen-1.5B。三条入口问题使用当时的可见历史，直接贪心生成逐条复现原答复。另有七条独立短问，涵盖中英翻译、只给一个动作、改写与修订、普通解释和不确定信息。两种直接作答中，一组原样输入，一组统一附加按本轮语言、数量和格式交付的要求。两种原生推理后续作答在同一道题上共用温度 0.6、top-p 0.95 采样得到的 128 词元以内前缀及相同的后续随机种子。到上限尚未结束时补上 `</think>`，分别比较后接零个或两个换行。可见答复都给 72 词元。结果只记录设定边界之后的文本和推理长度。 [多轮对话退化研究](https://arxiv.org/abs/2505.06120) · [Multi-IF](https://arxiv.org/abs/2410.15553)

| 解码 | 词面、格式和结束条件均通过 | 自然结束 | 作答生成耗时中位数 |
| --- | ---: | ---: | ---: |
| 直接贪心 | 2/10 | 7/10 | 0.99 秒 |
| 加入通用作答提醒 | 2/10 | 9/10 | 0.96 秒 |
| 原生推理后只补结束标记 | 0/10 | 2/10 | 4.69 秒 |
| 原生推理后补结束标记及换行 | 0/10 | 3/10 | 4.71 秒 |

十次推理都达到 128 词元上限，生成该段的耗时中位数约 8.08 秒。换行改变全部十条作答，但两种方案在这档推理上限均未通过。通用提醒让部分答复提前结束，翻译仍会增添与原句无关的内容，对快递到达日期还给出了错误的确定答复。实测约 289 秒，CUDA 张量峰值约 3.34 GiB。

人工核对首先发现改期题的自动判分偏差。部分禁词的单项集合在 Python 中被误写成字符串，导致逐字匹配，单动作题还漏认了「拿」「移」等正常动词。修复后按留存文本重新计算两组实验分数，并增加了检查评分器输入形状、有效改期和自然动词的单元测试。原始生成保持一致。上表及下表使用修正后的口径。

## 冻结底座上的末层学习

独立编写 32 个训练问答，翻译、单动作、改写和小事实各八条。其问题与十道评测题不重合，任务类型相同。两组各训练一个秩 16、49,152 参数的末层残差，共用相同的 297 个目标词元、采样顺序和 192 次参数更新。每步都用正确回复做交叉熵，其中两处优先从冻结底座最早偏离示范的答案位置抽样。实验组额外要求正确词元的分数高于原模型所选词元。底座和词表投影冻结。评测只在普通回答上安装残差，随后移除并核对能逐词复现冻结入口。 [ReFT](https://arxiv.org/abs/2404.03592) · [LoRA 遗忘对照](https://arxiv.org/abs/2405.09673)

| 生成 | 词面、格式和结束条件均通过 | 自然结束 |
| --- | ---: | ---: |
| 冻结底座 | 2/10 | 7/10 |
| 只用交叉熵 | 2/10 | 10/10 |
| 加首个分歧词元的比较 | 1/10 | 8/10 |

只用交叉熵的残差学会将入口的一道题缩成「把书桌里的废纸拿出来。」，但将原来答对的周日改期错改为周三。额外比较的候选有两条出现明显重复。两组仍无法完成入口的翻译，窗帘题分别复述了结论，遗漏解释。首次两组训练与生成约 133 秒，CUDA 张量峰值约 3.35 GiB。候选仅保存在本地研究目录。

独立进程重载两个残差后逐词复现全部 20 条候选答复，修正后的判分也与留存文本重算结果一致，用时约 118 秒。4 项评估回归和 49 项版次记忆单元测试通过。普通回复仍使用原有入口路径。

检查到 FollowBench 官方仓库有公开中文分区和具体要求分层。普通格式分区的作者评分还用到 GPT-4，适合挑出可由本机规则和独立人工判断的部分做留出。COIG-CQIA 的数据卡未明确给出整体许可，也未看到与这批改口、短答和现代中英翻译直接对应的单一分区。本轮训练用自编例子。后续优先扩大独立检验，再用可核验的成品监督普通回复，并把原模型已能答对的修订作为保持约束。 [固定版 FollowBench 中文数据](https://github.com/YJiangcm/FollowBench/tree/6278f4c1377b4eafab737267b8b21acd52ea0e52/data_zh) · [固定版 COIG-CQIA 数据卡](https://huggingface.co/datasets/m-a-p/COIG-CQIA/blob/8b55868c6168adf86c30e7ca0f782cca1c514297/README.md)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaSmoke = 'runs/spora-version-local-chat-audit-v4/result.json'
$sporaBoundary = 'runs/spora-local-language-boundary-fresh.json'
$sporaResidual = 'runs/spora-ordinary-reply-residual-fresh'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_local_language_boundary $sporaSmoke $sporaBoundary
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_ordinary_reply_residual $sporaSmoke $sporaBoundary $sporaResidual
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_ordinary_reply_residual $sporaSmoke $sporaBoundary $sporaResidual --verify
```
