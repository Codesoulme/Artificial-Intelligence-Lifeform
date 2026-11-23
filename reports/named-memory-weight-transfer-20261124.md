# DeepSeek 孢子后层权重迁移实验

此前的短程答案监督与蒸馏没有提高完整对话通过率。本轮直接迁移千问 2.5 1.5B 指令版的部分权重，测量普通中文生成与持久记忆回答能否同时保持。所有试验从同一 DeepSeek 孢子封装检查点加载，在原有孢子层中写入第 22 至 27 层权重。层号从零开始。两处读取层在第 5 与 17 层，最后一个记忆残差在第 21 层。

DeepSeek 官方将该尺寸的蒸馏版追溯到 Qwen2.5-Math-1.5B。数学版和指令版都把 Qwen2.5-1.5B 标为基础模型。两个发布版的层尺寸对齐，旋转位置频率和特殊词元定义有差异。实验统一使用 DeepSeek 的聊天格式、位置编码和输出层，结果仅适用于这个固定接口。 [DeepSeek 蒸馏模型](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) · [千问数学底座](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) · [千问指令版](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

## 直接插值

两套中文对话各含十组、二十一轮。每组把实际回复带入下一轮。记忆对照让模型用当前值和历史值回答四种新问法，共三十二题。

| 后六层改动 | 原题整组 / 逐轮 | 新题整组 / 逐轮 | 受控记忆逐题 |
|---|---:|---:|---:|
| 原封装候选 | 0/10 · 0/21 | 0/10 · 0/21 | 32/32 |
| 前馈权重混入 50% | 0/10 · 0/21 | 0/10 · 0/21 | 8/32 |
| 前馈权重全量替换 | 0/10 · 0/21 | 0/10 · 0/21 | 1/32 |
| 完整解码层权重混入 50% | 0/10 · 0/21 | 0/10 · 1/21 | 13/32 |
| 完整解码层权重全量替换 | 0/10 · 0/21 | 0/10 · 0/21 | 2/32 |
| 重新载入原后层权重 | 0/10 · 0/21 | 0/10 · 0/21 | 32/32 |

所有条件的个体神经状态摘要一致。受控记忆测试在生成前核对神经读取证据，三十二题的目标事实均与证据一致。混入后出现的答错发生在证据转成生成文本的过程中。恢复原后层权重后，两套对话和记忆评测的 74 条输出与开始时逐条一致。运行结果位于 `runs/named-memory-weight-transfer-v1/`。

## 指令微调权重差

以千问指令版权重减去其基础版权重，提取指令微调增量，按系数加到 DeepSeek 孢子后六层。方法来自任务向量研究。直接插值会同时改变两个发布模型的预训练差异，因此使用各自公开的基线来源核实这一比较。增量与加权在 FP16 计算，层结构继续使用孢子实现。 [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)

| 指令增量系数 | 原题整组 / 逐轮 | 新题整组 / 逐轮 | 受控记忆逐题 |
|---|---:|---:|---:|
| 0.15 | 0/10 · 0/21 | 0/10 · 0/21 | 32/32 |
| 0.50 | 0/10 · 0/21 | 0/10 · 0/21 | 32/32 |

同样改动最后六层，权重差在所测两档保住了记忆回答，中文得分未增加。所有改动只作用于实验进程中的权重。运行结果位于 `runs/named-memory-weight-task-vector-v1/`。

将相同的权重差扩展到 24 层，保留含记忆读取和残差的第 5、9、17、21 层。在 0.50 和 1.00 两档中，两套中文仍各为 0/10 组和 0/21 轮，受控记忆仍为 32/32。逐题结果位于 `runs/named-memory-weight-task-vector-v2/`。

## 对话格式复核

两个分词器的 151,643 个普通词元编号一致。将千问的三个对话标记映射至模型中未分配的词表行，同时写入对应的千问输入向量和输出向量。普通中文评测使用千问格式与空系统消息。记忆评测恢复这几个词表行，沿用 DeepSeek 格式。

仅切换对话格式，以及同时给 24 层加上完整指令增量，两套题各为 0/10 组和 0/21 轮，受控记忆均为 32/32。恢复原后层权重后，仍使用同一实验格式重跑，74 条输出逐条一致。逐题结果位于 `runs/named-memory-weight-chat-format-v1/`。

在上述 24 层和对话格式条件上，再为输入词嵌入和输出投影的 151,643 个普通词元叠加指令增量。两套题仍各为 0/10 组和 0/21 轮，受控记忆为 32/32。逐题检查可见复述、重复和遗漏用户更正。这个条件沿用 DeepSeek 位置编码，并保留四个记忆层计算。结果位于 `runs/named-memory-weight-vocab-v1/`。

## 同一指令底座的结构对照

完整千问指令版权重先用原生格式评测，再原位迁入孢子的 28 层统一投影、可学习位置表、两处前馈记忆读取及两处记忆残差。位置表从千问自己的旋转编码初始化。个体神经状态来自同一份 DeepSeek 孢子检查点，摘要一致。普通对话沿用千问的输入输出矩阵和结束条件。

| 条件 | 原题整组 / 逐轮 | 新题整组 / 逐轮 | 受控记忆 |
|---|---:|---:|---:|
| 原版千问指令版 | 5/10 · 13/21 | 4/10 · 8/21 | — |
| 迁入孢子结构，沿用原记忆回答层 | 5/10 · 13/21 | 4/10 · 8/21 | 6/32 |
| 用千问生成分数重新校准记忆回答层 | 5/10 · 13/21 | 4/10 · 8/21 | 32/32 |

迁入结构后，42 条普通对话回复有 41 条逐字一致。变化的一条指代题仍判为错误。只用 64 条已知地址和时间的训练请求校准记忆回答层，训练 192 步，底座与个体保持冻结。独立进程重新加载权重，复现迁入后的全部对话和校准后的全部记忆结果。原来的生成词表即使对齐普通词元，因底座输出分布不同，记忆回答层仍需适配。

这一对照说明，当前孢子结构允许同尺寸指令模型保留已测多轮表现，并用同一份神经状态完成受控记忆读取。自然姓名定位、时间选择和持续写入在千问配置上的效果尚待验证。运行记录及权重位于 `runs/named-memory-qwen-structure-control-v2/`。

## 全骨干插值

以校准后的千问孢子为起点，把 DeepSeek 权重按比例混入 28 层语言计算、末层归一化、普通词元输入和输出投影。固定千问的对话标记与位置表、孢子记忆模块及个体状态。每档均从原文件独立计算插值。

| DeepSeek 权重比例 | 原题整组 / 逐轮 | 新题整组 / 逐轮 | 记忆精确 / 含正确值 |
|---|---:|---:|---:|
| 0% | 5/10 · 13/21 | 4/10 · 8/21 | 32/32 · 32/32 |
| 15% | 0/10 · 1/21 | 0/10 · 0/21 | 28/32 · 32/32 |
| 35% | 0/10 · 0/21 | 0/10 · 0/21 | 18/32 · 31/32 |
| 65% | 0/10 · 0/21 | 0/10 · 0/21 | 23/32 · 32/32 |

恢复千问权重后复现全部 74 条输出，各档个体摘要相同。15% 已使完整对话得分从 9/20 降到 0/20。这组实验度量的是固定千问接口下直接线性插值的效果，未针对每档适配新的内部表示或对话格式。逐题记录位于 `runs/named-memory-backbone-mixture-v1/`。

## 复现

在 `D:\SPORA` 下执行，使用固定版本的千问底座权重。 [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

```powershell
$checkpoint = 'runs/named-memory-layer9-packed-reply-v1/packed.pt'
$donor = 'D:/SPORA-runtime/models/Qwen2.5-1.5B-Instruct-989aa798'
$ancestor = 'D:/SPORA-runtime/models/Qwen2.5-1.5B-8faed76/model.safetensors'
.\.venv\Scripts\python.exe -m scripts.audit_named_memory_weight_transfer --checkpoint $checkpoint --donor $donor --output runs/named-memory-weight-transfer-v1
.\.venv\Scripts\python.exe -m scripts.audit_named_memory_weight_transfer --checkpoint $checkpoint --donor $donor --ancestor $ancestor --output runs/named-memory-weight-task-vector-v1 --variants blocks-last-six-low blocks-last-six-half
.\.venv\Scripts\python.exe -m scripts.audit_named_memory_weight_transfer --checkpoint $checkpoint --donor $donor --ancestor $ancestor --output runs/named-memory-weight-task-vector-v2 --variants plain-blocks-half plain-blocks-full
.\.venv\Scripts\python.exe -m scripts.audit_named_memory_weight_transfer --checkpoint $checkpoint --donor $donor --ancestor $ancestor --output runs/named-memory-weight-chat-format-v1 --variants original plain-blocks-full --chat-markers
.\.venv\Scripts\python.exe -m scripts.audit_named_memory_weight_transfer --checkpoint $checkpoint --donor $donor --ancestor $ancestor --output runs/named-memory-weight-vocab-v1 --variants plain-blocks-and-vocab-full --chat-markers
$control = 'runs/named-memory-qwen-structure-control-v2/result.json'
.\.venv\Scripts\python.exe -m scripts.audit_named_memory_qwen_structure_control --checkpoint $checkpoint --donor $donor --output $control --updates 192
.\.venv\Scripts\python.exe -m scripts.audit_named_memory_qwen_structure_control --checkpoint $checkpoint --donor $donor --output runs/named-memory-qwen-structure-control-v2/reload.json --verify $control
.\.venv\Scripts\python.exe -m scripts.audit_named_memory_backbone_mixture --checkpoint $checkpoint --donor $donor --calibration $control --output runs/named-memory-backbone-mixture-v1
```
