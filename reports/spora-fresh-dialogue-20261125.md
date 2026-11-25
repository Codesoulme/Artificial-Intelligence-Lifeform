# 三至五轮中文对照与千问 3.5 本机试跑

上一轮两种千问 2.5 配置都答对了 6/8 道短改口题，在长格式题上多因生成上限中断。这轮用另外编写的自然对话查验，看看提高末轮预算后，DeepSeek 孢子和更小的新底座能否完成可用的成品。实验前固定 6 个场景、24 条用户消息以及每个场景的当前要求、已取消内容和交付标准。考察对象交接、条件变化、转题和信息不足。按 AFVE-Bench 的思路分别检查新要求、被取消的内容和成品可用性。 [短改口对照](spora-reply-backbone-control-20261125.md) · [AFVE-Bench](https://arxiv.org/abs/2611.10541) · [中文 Hy-MultiTurn](https://arxiv.org/abs/2607.29196)

## 设置

比较原版 DeepSeek-R1-Distill-Qwen-1.5B、现有 DeepSeek 孢子检查点、原版 Qwen3.5-0.8B。三者使用本机 CUDA FP16、各自的聊天格式、直接作答和贪心解码。前几轮最多生成 96 词元，末轮和人工整理的现行要求各给 224 词元。所有轮次记录模型自己的回复，原样带入下一轮，即使早期回复达到上限。两路 DeepSeek 均确认每次输入保留全部前文。千问输入直接传入完整消息列表。人工复核逐题判定是否能直接使用。人工整理输入只测回答上限，评分不计入分轮对话或孢子记忆。

千问 3.5 使用官方固定版本，核对权重与分词器 SHA-256。沿用本机 PyTorch 2.5.1 CUDA 12.1，将 Transformers 5.3.0 安装到单独目录，由千问试跑进程通过环境变量加载。GTX 1060 上 Gated DeltaNet 使用库自带的 PyTorch 实现。使用聊天结束符和模型配置指定的文本结束符作为停止条件。两路 DeepSeek 使用原环境和原检查点，运行后校验了个体张量和源文件摘要。两个结果的案例摘要相同，评测脚本摘要不同，仅因千问试跑前明确了这一组结束符，DeepSeek 生成与评分路径未受该改动影响。 [官方千问固定版本](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)

## 结果

| 指标 | DeepSeek 原版 | DeepSeek 孢子 | 千问 3.5 0.8B |
| --- | ---: | ---: | ---: |
| 分轮末答可直接用 | 0/6 | 0/6 | 1/6 |
| 给出整理后的现行要求，可直接用 | 0/6 | 0/6 | 2/6 |
| 分轮末答正常结束 | 6/6 | 6/6 | 5/6 |
| 全部分轮答复正常结束 | 23/24 | 23/24 | 10/24 |
| 分轮末答生成时间中位数 | 3.38 秒 | 3.96 秒 | 4.93 秒 |

| 场景 | DeepSeek 两版末答 | 千问 3.5 末答 |
| --- | --- | --- |
| 相馆取件 | 空泛确认，未交付取件消息 | 保留河东店，正确改为星期五傍晚取一张，可直接发 |
| 缺失的送达时间 | 泛泛建议联网，未回答能否确定三点前收到 | 提到无法确定，夹杂未证实的处理时程，核实步骤截断 |
| 小组分工 | 保留旧集合时间与旧负责人 | 同时写出两个集合时间，发卡职责仍错 |
| 从植物转题 | 说到了冷凝，未交代水来自空气，也未按要求分两句 | 错把水的来源说成冰杯里的水 |
| 文件交接 | 将已通过的复核当成待办，反复提预算表 | 提到了正确时间和 PDF，仍提预算表，并让收件人确认后发送 |
| 雨天集合 | 选对地点，仍要求带地图，还凭空补了日期 | 选对地点和时间，仍要求带地图，额外要求提前十分钟 |

原版 DeepSeek 与孢子末答 6/6 逐字相同，最长 87 词元。这批直接作答失败远早于 224 词元上限。千问在人工整理现行要求后写对了取件和分工两题，分工的差异支持后续单独测对话状态。它在单独的硬件冒烟题中仍答错冰块融化后的水温变化。千问本轮 14/24 条前后轮答复在预算内未正常结束，下一轮不能用这一结果单独归因于模型架构。整轮 CUDA 张量峰值分别为 DeepSeek 两路合计 4.94 GiB、千问 1.67 GiB。墙钟时间分别约 357 秒和 245 秒，含加载和校验。

本轮不安排千问 3.5 结构迁移。它对短通知有可见收益，但在当前解码方式下还没改善两人分工和多次更正。后续用冻结的 DeepSeek 孢子分两路诊断：用相同题目给出人工确认的现行要求，测直接生成和短成品示范各能补回多少；用希望保留的个体张量在新题上交换当前字段与旧字段，检验读出能否服务普通对话。以当前要求、废弃内容和可用成品三项分别计分。小样本作为选路线的依据，开放中文和重启后记忆仍需独立联合验收。 [StateMemBench](https://arxiv.org/abs/2608.19652)

## 复现

原本机 DeepSeek 和孢子检查点沿用开放中文对照的准备步骤。千问从固定官方版本取 `config.json`、`chat_template.jinja`、`tokenizer_config.json`、`tokenizer.json`、`model.safetensors.index.json` 和唯一的权重分片，保存到独立模型目录。脚本校验权重和分词器。本机已有 `tokenizers==0.22.1` 和 `huggingface-hub==1.33.0`。 [原环境说明](spora-open-chinese-20261125.md)

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaRun = 'runs/spora-fresh-dialogue-new'
$sporaDeps = 'D:/SPORA-runtime/qwen35-transformers-5.3.0'
$sporaWeights = 'D:/SPORA-runtime/models/Qwen3.5-0.8B-2fc06364'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_fresh_dialogue deepseek "$sporaRun/deepseek.json"
.\.venv\Scripts\python.exe -m pip install --target $sporaDeps --no-deps transformers==5.3.0 accelerate==1.12.0 safetensors==0.7.0 pillow==12.1.1 regex==2026.2.28 pyyaml==6.0.3
$env:PYTHONPATH = $sporaDeps
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_fresh_dialogue qwen35 "$sporaRun/qwen35.json" --weights $sporaWeights
```

逐轮输出位于 `runs/spora-fresh-dialogue-v1/deepseek.json` 与 `runs/spora-fresh-dialogue-v1/qwen35.json`。
