# DeepSeek 孢子中文适配的层分布对照

从采用 DeepSeek-R1-Distill-Qwen-1.5B 的同一个孢子检查点出发，冻结底座及已保存的个体。在普通对话路径上比较低秩残差的位置和宽度。模型继续使用自己的前轮回复生成下一轮。宽覆盖组在 28 层中选用 24 层，第 5、9、17、21 层已有记忆读取或残差，因此这四层未添加对话残差。

三组使用同一个 905 条请求池、相同的采样顺序和 256 次更新。请求池包含已有中文基础素材、更正练习、审阅过的回答和项目自编的多轮素材。三组实际抽到 202 条请求，监督了 2,036 个不同的答案位置，累计 2,508 次。各组的学习率为 `5e-5`，每次最多投影 16 个答案位置。测试题对照的是层分布和参数规模在这一训练预算下的效果。固定学习率的设置未单独校准不同秩和不同层分布的更新幅度。低秩适配文献提供冻结底座的实验依据，孢子使用的是带 GELU 的独立残差。 [LoRA](https://arxiv.org/abs/2106.09685) · [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)

| 对话残差 | 可训练参数 | 短多轮 A，整组 / 逐轮 | 短多轮 B，整组 / 逐轮 | 受控持久读取 |
| --- | ---: | ---: | ---: | ---: |
| 5 层，秩 8 | 122,880 | 0/10 · 6/21 | 0/10 · 7/21 | 32/32 |
| 5 层，秩 37 | 568,320 | 1/10 · 8/21 | 0/10 · 7/21 | 32/32 |
| 24 层，秩 8 | 589,824 | 2/10 · 10/21 | 3/10 · 11/21 | 32/32 |

三份检查点的全部原始张量逐元素对齐父检查点，仅新增各自的普通对话残差。24 层组的可训练参数比 5 层秩 37 组多 3.8%，两套短题整组合计多通过 4 组。提高 5 层残差的秩也改善了第一套题的部分回复。单种子、两套已用过的短诊断题和约 3.8% 的参数差异限制了归因范围。

## 开放对话开发复核

独立进程逐个重载模型，按 `conversation-development-v5.json` 中的逐轮标准审查自然续聊。每轮上限 96 词元。按标准审阅原文，要求三轮都符合语义和格式，且在预算内结束。该开发集有 6 组、18 轮，曾用于早期候选的诊断。18 条问题与本次训练源的 662 条不同用户问题无逐字重合。

| 重载条件 | 完整组 | 自然结束 |
| --- | ---: | ---: |
| 共同父检查点 | 0/6 | 15/18 |
| 5 层，秩 37 | 1/6 | 17/18 |
| 24 层，秩 8 | 1/6 | 16/18 |
| 较新的三处 GELU 门控检查点 | 0/6 | 18/18 |
| 三处门控检查点增加 19 层残差 | 0/6 | 17/18 |

两个训练组均通过影子长度题。24 层组能问对店员应该确认的尺寸，但在下一轮复述了询问意图，没写出直接问句。两组均未执行取件时间提前半小时的要求，也未在最后一轮保留两人的最新分工。短题上的层分布收益尚未扩展到此次开放对话。逐轮原文位于 `runs/spora-dialogue-placement-audit-v1/`。

## 在三处门控迁移版本上复测

在三处 GELU 门控迁移检查点上为剩余 19 个兼容层加入秩 8 残差。冻结已有的五处普通对话残差。本轮使用同一请求池，以 `3e-5` 训练 384 步，实际抽到 280 条请求和 3,739 次答案位置曝光。两套短题从 4/10、5/10 降到 3/10、4/10，受控持久读取为 32/32。开放复核中，第三轮仍答错了取件时段和最终分工。此候选仅用于诊断层覆盖在不同起点上的效果。

保存权重后，第一次独立评测遇到 Windows 虚拟内存不足及 CUDA 驱动异常。将输入词表一同放在 CPU 的重试触发了结构层随输入词表移到 CPU，造成额外峰值。独立评测现在可以通过 `--stage-output-projection` 单独延后输出词表投影的显存分配。该方式完成了 18 轮自由生成和逐轮记录。检查点存于 `runs/spora-gelu24-dialogue-placement-v1/`。

## 复现

在 `D:/SPORA` 下逐组运行。这些训练使用相同父检查点和教材。输出目录存有逐题回复与检查点摘要。

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$sporaCommon = @('--checkpoint', 'runs/named-memory-layer9-packed-reply-v1/packed.pt', '--steps', '256', '--varied', '--composition', '--dialogue-data', 'data/dialogue-everyday-training-v1.json', '--learning-rate', '5e-5', '--loss-tokens', '16', '--fresh-holdout', '--stage-embeddings')
$sporaWide = @(0,1,2,3,4,6,7,8,10,11,12,13,14,15,16,18,19,20,22,23,24,25,26,27)
$sporaNarrow = @(12,19,22,24,27)
.\.venv\Scripts\python.exe -u -m scripts.train_named_memory_dialogue @sporaCommon --layers $sporaWide --rank 8 --output runs/spora-all-layer-dialogue-pilot-v1
.\.venv\Scripts\python.exe -u -m scripts.train_named_memory_dialogue @sporaCommon --layers $sporaNarrow --rank 8 --output runs/spora-five-layer-dialogue-pilot-v1
.\.venv\Scripts\python.exe -u -m scripts.train_named_memory_dialogue @sporaCommon --layers $sporaNarrow --rank 37 --output runs/spora-five-layer-rank37-dialogue-pilot-v1

$sporaOpen = @('--development-only', 'data/conversation-development-v5.json')
.\.venv\Scripts\python.exe -m scripts.audit_named_dialogue_composition runs/named-memory-layer9-packed-reply-v1/packed.pt runs/spora-dialogue-placement-audit-v1/parent.json @sporaOpen
.\.venv\Scripts\python.exe -m scripts.audit_named_dialogue_composition runs/spora-all-layer-dialogue-pilot-v1/integrated.pt runs/spora-dialogue-placement-audit-v1/distributed.json @sporaOpen
.\.venv\Scripts\python.exe -m scripts.audit_named_dialogue_composition runs/spora-five-layer-rank37-dialogue-pilot-v1/integrated.pt runs/spora-dialogue-placement-audit-v1/concentrated.json @sporaOpen
.\.venv\Scripts\python.exe -m scripts.audit_named_dialogue_composition runs/spora-composition-late-gates-20-23-24-v1/candidate.pt runs/spora-dialogue-placement-audit-v1/previous.json @sporaOpen

$sporaExtra = @(0,1,2,3,4,6,7,8,10,11,13,14,15,16,18,20,23,25,26)
.\.venv\Scripts\python.exe -u -m scripts.train_named_memory_dialogue --checkpoint runs/spora-composition-late-gates-20-23-24-v1/candidate.pt --output runs/spora-gelu24-dialogue-placement-v1 --steps 384 --varied --composition --dialogue-data data/dialogue-everyday-training-v1.json --layers $sporaExtra --rank 8 --learning-rate 3e-5 --loss-tokens 16 --fresh-holdout --stage-embeddings
.\.venv\Scripts\python.exe -m scripts.audit_named_dialogue_composition runs/spora-gelu24-dialogue-placement-v1/integrated.pt runs/spora-dialogue-placement-audit-v1/current-expanded.json @sporaOpen --stage-output-projection
```
