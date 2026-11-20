# 清醒版孢子的语言骨干对照

在本机读取 Windows 入口使用的 DeepSeek 权重与孢子集成检查点。在关闭记忆读取的情况下，同一实例切换第 6、12 层，保留相同分词、提示、生成缓存和 128 词元上限。六道题来自上一轮已查看的普通聊天诊断。

| 骨干 | 参考答案 NLL／词元 | 自然结束 | 与原始层逐词元一致 |
|---|---:|---:|---:|
| 原始 DeepSeek | 3.4084 | 4/6 | 6/6 |
| 替换第 6 层 | 3.3675 | 5/6 | 1/6 |
| 替换第 12 层 | 3.4991 | 5/6 | 0/6 |
| 同时替换第 6、12 层 | 3.4098 | 4/6 | 0/6 |

四个组合都算对了冰块数。原始 DeepSeek 回避热水问题，双层替换输出了错误解释。第 12 层单独替换把翻译输出改为英文问句，但漏译「最」。同样的平均 NLL 和结束数可以对应不同的答案错误。

## 用学生生成前缀校正完整替换层

冻结其余权重，训练第 6、12 层的归一化参数和前馈输出偏置。用 12 条另外编写的训练问题，先采集学生生成的 16 词元以内续写，再由原始 DeepSeek 给同一前缀打分。训练同时用教师分布和人工短答案。前缀采集只做一次，参考 GKD 在学生访问过的序列上对齐教师分布的做法。[GKD 论文](https://arxiv.org/abs/2306.13649)

| 校正条件 | 六道开发题平均参考 NLL | 教师 KL | 自然结束 |
|---|---:|---:|---:|
| 校正前 | 3.0880 | 0.2915 | 3/6 |
| 32 步，学习率 0.0003，答案权重 0.2 | 3.0765 | 0.2874 | 4/6 |
| 64 步，学习率 0.003，答案权重 2 | 2.5382 | 0.4384 | 6/6 |

两种设置均使用 96 词元生成上限。第二组同一时间改了步数、学习率和答案权重。较强校正让六道开发题全部结束。在另六道普通问题上，结束数从 6/6 降至 5/6，翻译及金属导热仍有错。七项持久记忆核对在校正前后均读出了指定值，包括历史值和模型名称，其中四条完整回答逐字一致。读取验证使用固定检查点内的三名人物和一个个体名称，覆盖当前值和一条旧值。默认聊天入口继续使用原检查点。

## 只替换第 6 层前馈网络

保留第 6 层 Qwen 注意力与位置编码，使用孢子第 6 层的 GELU 前馈网络作为初始化。第 12 层保留完整孢子替换块。用项目中文教材获取 3,483 个训练激活和从另外八段教材取得的 369 个留出激活。第一轮拟合原始前馈网络的输出。第二轮从第一轮权重出发，再对齐整网输出的下一词元分布。

| 配置 | 第 6 层留出激活相对平方误差 | 普通题教师 KL | 96 词元内结束 |
|---|---:|---:|---:|
| 孢子前馈初始化 | 0.873 | — | — |
| 128 步局部训练 | 0.616 | 0.2351 | 4/6 |
| 再做 48 步整网输出对齐 | 0.747 | 0.2235 | 2/6 |
| 当前双层替换 | — | 0.2915 | 3/6 |
| 第 6 层恢复原始结构，第 12 层继续用孢子 | — | 0.1701 | 3/6 |

局部拟合降低了激活误差，但模型仍会错译、重复或回避普通问题。进一步对齐整网输出后，KL 继续降低，结束率反而下降。本轮使用手写小样本和单一种子，普通题仅支持开发诊断。可复现的候选权重保存在 `runs/awake-replacement-replay-v1/`、`runs/awake-replacement-replay-v2/`、`runs/awake-partial-layer-v1/` 和 `runs/awake-partial-layer-v2/`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_language_layers.py --output runs/layer-audit.json
.venv/Scripts/python.exe scripts/train_awake_replacement_replay.py --output runs/replay --steps 64 --learning-rate 0.003 --reference-weight 2
.venv/Scripts/python.exe scripts/train_awake_partial_layer.py --output runs/partial
.venv/Scripts/python.exe scripts/train_awake_partial_layer.py --output runs/partial-output --steps 0 --output-steps 48 --start runs/partial/partial-layer.pt
```

## 复用预训练投影，把记忆读入前馈网络

借鉴 Net2Net 保留初始函数再改结构的做法，将预训练的门控和上投影合成一个线性映射，保留 SiLU 与下投影。将原先在层末加入的孢子记忆残差按设定比例移到该层前馈输入。第 6、12 层替换块沿用同一集成检查点。[Net2Net 论文](https://arxiv.org/abs/1511.05641)

第 5 层和第 17 层的单层实验中，三条关闭记忆的普通题在改动前后逐词元一致。仅融合投影时，两条问句分别在以咖啡、水为事实的状态下正确生成，四个条件均与原路径一致。将读取全部移入前馈入口后，两层分别都失去全部四个正确输出。第 5 层将 5% 或 15% 的残差移入前馈入口时，保留四个正确值。比例 30% 时保留三个。

固定第 5 层 15% 后，另用两个地址、各两次写入测试当前值和历史值。原路径和新路径均正确生成 3/4 项。两边都漏读了阿宁的历史值「果汁」，并给出不同的无关介绍。本轮使用实验直接指定的地址和版本。新计算仍依赖原有层末读取，大范围迁移需重新对齐读取位置。运行文件不覆盖正式检查点。

将测试扩至四个地址各两次写入。八项当前值和历史值的矩阵读数均正确。原路径、第 5 层 5% 迁移和 15% 迁移各正确生成七项。启用旧的首词元输出适配器后，原路径与 15% 迁移均降为五项，其中三条答成了「水」。漏读的「果汁」在矩阵中已经正确，差异发生在读数进入生成后。

原生读取器现用批量矩阵乘法一次计算一页的全部词元查询，并复用读取结果计算页选择分数。GTX 1060 Max-Q 的读取器微基准使用批量 4、384 词元、3 页及宽度 90，30 次调用平均从 12.69 毫秒降到 7.06 毫秒，峰值额外分配从 157.63 MiB 降到 105.59 MiB。新旧 FP16 输出的最大绝对差为 0.001953。单条 384 词元输入的显存峰值基本一致。真实模型复测时，15% 路径仍通过原四个换值条件和八项历史中的七项。56 项相关单元测试通过。

改用已有、训练覆盖八种饮品的适配器，原位置与 15% 迁移路径均正确生成八项中的八项。旧适配器训练覆盖两种饮品。两份权重的训练步数和种子也不同，这组实验无法分离数据覆盖的独立影响。读取器仍由实验给定人物和版本，且全部输出属于已有的有限词表类别。真实模型在 CPU 持有孢子个体、GPU 执行生成时遇到过设备不匹配，适配器现将读数传到自身权重的设备，并通过跨设备回归测试。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_feedforward_memory.py --layer 5 --fraction 0.15 --output runs/feedforward-memory.json
```

## 替换层参数的损坏恢复

把第二轮校正权重加载到临时推理实例，记录三条普通回答、三项当前或历史事实和孢子的名称。只在第 6、12 层的归一化参数与前馈输出偏置中注入损坏，共 15,360 个受测参数。将预先标明的 10%、50% 或 80% 参数置零，对比直接回到校正前权重、用滑动平均影子填回受损位置、用精确快照填回受损位置。影子由校正前后的参数各占一半生成。每份快照约 63 KiB。修复范围由实验提供的布尔掩码指定。

| 损坏比例 | 损坏后普通回答一致 | 回滚到校正前 | 滑动平均影子 | 精确快照 |
|---|---:|---:|---:|---:|
| 10% | 1/3 | 1/3 | 3/3 | 3/3 |
| 50% | 0/3 | 1/3 | 2/3 | 3/3 |
| 80% | 0/3 | 1/3 | 2/3 | 3/3 |

两种影子均保住四项事实与身份检查。精确快照在三档均还原了受测参数的完整摘要，并通过同题验收。滑动平均影子在 50% 和 80% 时被普通回答一致性门槛拒绝。验收器现在也拒绝非有限指标，显式掩码能避免把合法的零值当成损坏。34 项相关单元测试通过。

实验验证了已知损坏位置的局部恢复。损坏检测、自动发布、全模型恢复和重新训练成本仍待验证。模型、个体名称以及记忆数据库摘要在实验前后核对通过。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_dna_recovery.py --delta runs/replay/delta.pt --profile runs/spora-awake-local --output runs/dna-recovery
```

## 单层和多层替换的中文对照

在同一个推理实例里依次切换原版、第 6 层、第 12 层、双层和第 18 层完整替换候选。关闭记忆读入。八组新增手写对话每组两轮，第二轮输入第一轮的真实生成，统一使用贪心解码，每轮最多 24 词元。严格评分要求内容和指定的短答格式一起正确。另用 C-Eval 官方验证集的高中语文、逻辑与初中数学作为外部对照。给模型题干和选项，从 A、B、C、D 的下一词元分数中选择最高分。

| 骨干 | 首轮严格正确 / 8 | 两轮严格正确 / 8 | 答到结束词元 / 16 | 三门公开题 | 所有骨干共有的短题 |
|---|---:|---:|---:|---:|---:|
| 原版 | 2 | 1 | 8 | 24/60 | 8/19 |
| 第 6 层 | 1 | 1 | 10 | 22/60 | 8/19 |
| 第 12 层 | 0 | 0 | 14 | 20/60 | 6/19 |
| 第 6、12 层 | 0 | 0 | 10 | 18/60 | 8/19 |
| 第 6、12、18 层 | 0 | 0 | 13 | 6/19 | 6/19 |
| 第 6 层仅替换前馈，第 12 层完整替换 | — | — | — | 19/60 | 7/19 |

双层相对原版改对 9 道公开题，也改错了 15 道。第 18 层候选最多接收 128 词元，只覆盖 19 道短题，其中 15 道是数学。在完全相同的 19 道题上，加入该层后由 8 道降到 6 道。短对话的格式与截断都会影响严格命中数，例如原版回答对了纽扣所在的盒子，多写了内容而失分。第 12 层有 14 次自然结束，两轮严格正确依然为零。两组实验都使用预先结束推理标记的提示方式，覆盖闭卷选择题和有固定答案的短对话。自由生成、开推理模式及记忆保持仍需直接验证。

公开题固定使用修订 `617524a00b307ff6f9933702f724131fe12ca7ce`。复现时取三个科目的 `val-00000-of-00001.parquet`，分别命名为 `high_school_chinese.parquet`、`logic.parquet` 和 `middle_school_mathematics.parquet`。需要 `pyarrow` 读取本地文件。逐题答案、输入长度和来源摘要保存在 `runs/awake-multiturn-layers-v1.json` 和 `runs/awake-ceval-layers-v2.json`。两轮运行分别耗时约 143 秒、109 秒，峰值显存分别约 3.95 GB、4.05 GB。[C-Eval 验证集](https://huggingface.co/datasets/ceval/ceval-exam/tree/617524a00b307ff6f9933702f724131fe12ca7ce)

```powershell
.venv/Scripts/python.exe scripts/audit_awake_multiturn_layers.py --output runs/awake-multiturn-layers-v1.json
.venv/Scripts/python.exe scripts/audit_awake_multiturn_layers.py --ceval runs/ceval-validation-617524a --partial runs/awake-partial-layer-v1/partial-layer.pt --output runs/awake-ceval-layers-v2.json
```

## 保留第 12 层注意力，只蒸馏前馈网络

第 6 层仍用完整孢子块，用第 12 层已有的孢子前馈权重初始化，在 3,483 个教师激活上训练 128 步。369 个留出激活上的相对平方误差从 0.952 降到 0.663。用同一批公开题和八项受控的现值、旧值查询，比较仅有第 6 层改造、现行双层改造和新的局部替换。

| 配置 | C-Eval | 记忆查询含预期值 |
|---|---:|---:|
| 第 6 层完整替换，第 12 层原版 | 22/60 | 8/8 |
| 第 6、12 层完整替换 | 18/60 | 7/8 |
| 第 6 层完整替换，第 12 层只改前馈 | 17/60 | 2/8 |

记忆来自预先写入的四个人物，各查询一个现值和一个旧值。实验指定地址和版本，统计回答是否含预期饮品。局部拟合误差和六道开发题的教师 KL 都下降，仍出现明显的事实生成退化。该候选保存在本地实验路径。逐题结果保存在 `runs/awake-ceval-layer12-partial-v1.json`，联合评测耗时约 120 秒。回归应同时覆盖语言、事实和历史值。

```powershell
.venv/Scripts/python.exe scripts/train_awake_partial_layer.py --layer 12 --steps 128 --output runs/awake-partial-layer12-v1
.venv/Scripts/python.exe scripts/audit_awake_multiturn_layers.py --ceval runs/ceval-validation-617524a --partial runs/awake-partial-layer12-v1/partial-layer.pt --output runs/awake-ceval-layer12-partial-v1.json
```

## 第 18 层从预训练权重开始变更结构

将原先分立的 Q、K、V 投影合成孢子用的单个 QKV 投影。等宽候选将每个 K、V 头复制到原来与之配对的六个 Q 头，保留原始旋转位置编码、归一化及输出投影。复用权重后无需训练即可运行。另一个候选保留原来的分组尺寸，同时合并前馈的两个输入投影。这一版保持原注意力的参数量，适合继续作为迁移起点。

| 第 18 层候选 | 普通题生成逐词元一致 | 八项记忆生成含预期值 | 注意力参数 | 首步分数的最大绝对差 |
|---|---:|---:|---:|---:|
| 等宽 QKV | 8/8 | 7/8，基线也是 7/8 | 9,441,792 | 0.02051 |
| 分组 QKV 与合并前馈 | 8/8 | 7/8，基线也是 7/8 | 5,507,072 | 0.02100 |

八道普通题取两轮评测中的首轮，生成上限 32 词元，比较前缀分数及带生成缓存的输出。记忆查询沿用四个人物的受控地址与版本。两种候选仍使用原有 SwiGLU 和旋转位置编码。脚本在临时推理实例上执行改造，检查源模型和个体检查点的文件摘要。逐题结果保存在 `runs/awake-attention-bridge-layer18-v1.json` 和 `runs/awake-attention-feedforward-layer18-v1.json`。

MOHAWK 分别对齐序列混合、块输出和模型预测，先从计算差异最大的组件入手。这为从复制投影继续迁移位置与注意力计算提供了参照。保留原结构的分组候选提供起始对照，后续需要同场测量绝对位置替换的代价和事实保持。[MOHAWK 论文](https://arxiv.org/abs/2408.10189)

```powershell
.venv/Scripts/python.exe scripts/audit_awake_attention_bridge.py --output runs/awake-attention-bridge-layer18-v1.json
.venv/Scripts/python.exe scripts/audit_awake_attention_bridge.py --keep-groups --fuse-feedforward --output runs/awake-attention-feedforward-layer18-v1.json
```

## 将合并投影扩展到其余 26 层

26 个原生 Qwen2 块都换成孢子的合并投影实现。第 6、12 层仍使用已经蒸馏过的孢子块，记忆读取层则继续使用原来的矩阵和读取接口。整个模型的参数数量保持一致。这一步保留 Qwen2 的分组、旋转位置编码、归一化和 SwiGLU 算法，启动时从已校验的原始权重重建投影。

与现行双层版相比，八道普通题的生成全都逐词元一致，八项事实输出均一致且七项答对，C-Eval 60 道题的选项均一致且答对 18 题。首步分数最大绝对差为 0.05225。首次聊天验证发现 Transformers 按解码层类型采集路由需要的隐藏表示，替换后该字段缺失。新块保留 Qwen2 的输出采集接口，修复后再次完成 GPU 对照及聊天验证。18 项集成单元测试通过，其中新增一项检查隐藏表示、有记忆的生成和重复加载。

实机用独立个体副本验证了名字「孢子」、用 `/记住` 写入米香玄米茶、点名读回饮品、追问职业和睡眠回放。重新启动后，个体名称与新事实仍能读回，睡眠核对一致，记忆文件摘要通过校验。一个把历史短语放在句首的问法仍要求明确对象。逐题结果位于 `runs/awake-migrated-backbone-v2.json`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_attention_bridge.py --all --ceval runs/ceval-validation-617524a --output runs/awake-migrated-backbone-v2.json
.\spora_chat.cmd --migrate-backbone
```

## 第 18 层迁移到学习型绝对位置

从预训练 QKV 开始，逐步降低旋转位置分量，学习一个 256 位置的绝对嵌入表和注意力输出投影。部署候选的旋转位置分量为零。两轮各训练 64 步，第二轮加入 24 条由其他六个地址和写入版本构造的记忆激活。下表的注意力误差来自八段留出文本。普通题仍是开发用的八组两轮短问答。记忆核对使用独立的四个地址。

| 第 18 层 | 留出注意力相对平方误差 | 严格答对的回合 / 16 | 八项记忆正确 |
|---|---:|---:|---:|
| 原始第 18 层 | — | 0 | 7/8 |
| 仅普通文本蒸馏 | 0.397 | 1 | 6/8 |
| 加入受控记忆回放 | 0.407 | 1 | 7/8 |

零初始化且直接停用旋转位置时，留出注意力误差是 0.440。局部误差较低的候选在查询中反而少答对一项。两轮的采样顺序有差异，结果仍需更多独立查询和更长位置验证。带回放的候选训练及验证耗时约 106 秒，本机产物和分题输出位于 `runs/awake-absolute-attention-memory-v1/`。

```powershell
.venv/Scripts/python.exe scripts/train_awake_absolute_attention.py --memory-replay --output runs/awake-absolute-attention-memory-v1
```

从新进程读取带记忆回放的候选，核对了文件、底座和层号。C-Eval 共 57 题落在 256 词元范围内，原第 18 层答对 17 题，换成绝对位置候选后答对 16 题。八项受控记忆均为 7/8，八道普通题中有三道生成逐词元一致。结果见 `runs/awake-absolute-attention-reload-v1.json`。

## 会话承接与解码对照

本地普通对话分支现在把真实生成的上轮回答带入下一轮，最多保留四轮。超长输入会提示分段。事实查询仍通过个体记忆分支。实机输入「写一句早安问候」后再要求改成简短的晚安，第二轮确认带入了一轮会话，但生成继续重复早安内容。相同测试中，身份、明确写入、点名读回、隐式改问职业和睡眠回放均通过。单独测试「你好」时，现有分流仍会询问用户想查哪条记录。

在 26 层合并投影的版本上，比较同一批八组短对话的解码策略。贪心解码给每轮 64 词元时有 16/16 回合自然结束。禁止重复四元组后为 15/16。打开推理并采样，每轮给 192 词元时为 14/16，两个回合在预算内没有给出最终回答。三组按短答格式严格计分均为 0/16，个别回答含有正确内容但仍违反格式。生成时保留了每组自己的真实上轮回答，结果保存在 `runs/awake-migrated-decoding-v1.json`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_multiturn_layers.py --decoding --output runs/awake-migrated-decoding-v1.json
```

## 绝对位置候选的聊天验证

在隔离个体上将第 18 层的学习型绝对位置注意力与 26 层合并投影一起加载。实机验证了名称、明确写入、点名读回、同人物改问和睡眠核对。重启后，在同一份记忆中读回现值「米香玄米茶」与上一版「青杏冰茶」，历史限定语放在姓名前后均得到相同结果，读取前后文件摘要一致。此候选使用 256 词元位置上限，超长消息提示分段。问候由可选普通对话分类器处理。

对普通分流分类器加入问候、改写和想起旧记录的对照。在新增的 8 条普通话语上，基础分流通过 3 条，旧分类器和新增样本训练版均通过 8 条。8 条记忆对照的误放分别为 2、3 和 3 条。扩充样本未改善分流，本机验证使用旧分类器。在实机多轮实验里，普通分支能接收上文，但生成仍可能重复问候或偏离改写要求。

## 共享位置编码迁入参数表

把运行时计算的旋转位置系数预先初始化为 768 个位置的可训练参数表。查询与键沿用预训练的旋转计算，参数表替代实时三角函数，共新增 98,304 个参数。聊天推理时冻结该表。与现行双层版相比，合并其余 26 层并启用参数表后，60 道 C-Eval 的选项全相同，8 道普通生成逐词元一致，受控记忆仍为 7/8。实机验证了名称、写入、读回、追问、旧值和睡眠核对。18 项集成测试通过。

逐题记录在 `runs/awake-migrated-position-table-v1.json`。本轮验证了位置系数从公式迁入模型参数的初始行为，尚未验证参数表经训练后的能力变化。

## 两个完整孢子块的多轮输出校正

固定第 6、12 层的孢子结构，只更新归一化参数与前馈输出偏置，共 15,360 个参数。使用项目中文教材的 166 个目标回合，保留每个回合原有的对话前文。训练同时拟合参考回答和原始 DeepSeek 在相同前缀上的分布。每三步加一次记忆回放，24 个候选前缀来自独立的六个人物地址及其当前、历史状态。回放目标取未经校正的双层版实际生成，用于保持现有行为。两轮各训练 72 步，采样顺序固定，唯一改变的是教师分布损失权重。

| 条件 | C-Eval | 短对话严格正确 | 短对话自然结束 | 受控记忆 |
|---|---:|---:|---:|---:|
| 校正前 | 18/60 | 0/16 | 14/16 | 7/8 |
| 教师权重 0.5 | 16/60 | 2/16 | 15/16 | 7/8 |
| 教师权重 2 | 17/60 | 2/16 | 14/16 | 7/8 |

短对话每轮最多生成 48 词元，评分要求答案与指定短答格式同时正确。受控记忆由评测指定人物地址和版本。两份增量权重在独立进程中重新加载，再通过正常的事实路由查询固定数据库。两个候选均保住七项持久记忆，教师权重 2 的版本七项完整回答均与校正前一致，权重 0.5 的版本有四项一致。独立进程中六道普通题的自然结束数从 6/6 降到 5/6。数据库摘要通过核对。

两轮含评测分别用时约 153 秒和 157 秒，峰值显存约 4.11 GB。两个候选均保留为实验权重，正式入口仍加载已验证的原权重。结果表明小范围输出校正能改变部分短答行为，恢复更广的中文能力仍需后续实验。逐题结果和增量分别位于 `runs/awake-dialogue-recovery-v1/`、`runs/awake-dialogue-recovery-v2/` 及各自的 `-runtime.json`。

```powershell
.venv/Scripts/python.exe scripts/train_awake_dialogue_recovery.py --ceval runs/ceval-validation-617524a --output runs/awake-dialogue-recovery-v1
.venv/Scripts/python.exe scripts/train_awake_dialogue_recovery.py --teacher-weight 2 --ceval runs/ceval-validation-617524a --output runs/awake-dialogue-recovery-v2
.venv/Scripts/python.exe scripts/audit_awake_replacement_memory.py runs/awake-dialogue-recovery-v2/delta.pt --profile runs/spora-awake-local --ordinary-head runs/deepseek-awake-ordinary-intent.pt --output runs/awake-dialogue-recovery-v2-runtime.json
```

## 第 18 层的归一化与前馈迁移

固定现行第 6、12 层和第 18 层的预训练注意力，依次替换第 18 层的两个归一化模块与前馈网络。用 81 段项目教材的激活拟合，在另外 8 段留出文本报告局部误差。LayerNorm 逐通道拟合缩放与偏置，两个位置的留出相对平方误差分别为 0.00097 和 0.00108。GELU 前馈复用旧第 18 层孢子实验的初始化，训练 256 步后，同一指标从 1.023 降到 0.609。

| 第 18 层配置 | C-Eval | 两轮短对话严格正确 | 自然结束 | 受控记忆 |
|---|---:|---:|---:|---:|
| 原预训练配置 | 18/60 | 0/16 | 14/16 | 7/8 |
| 仅注意力前换 LayerNorm | 17/60 | 0/16 | 14/16 | 7/8 |
| 仅前馈前换 LayerNorm | 19/60 | 0/16 | 14/16 | 7/8 |
| 两处 LayerNorm | 19/60 | 0/16 | 14/16 | 7/8 |
| 仅 GELU 前馈 | 15/60 | 0/16 | 13/16 | 0/8 |
| 两处 LayerNorm 与 GELU 前馈 | 15/60 | 0/16 | 13/16 | 0/8 |

公开题使用同一批题及解码规则。短对话每轮最多 48 词元。记忆实验直接指定地址和版本，统计八项现值与旧值中，答案包含预期值的项数。GELU 的局部拟合改善未转化为整网记忆保持。两处 LayerNorm 作为可选分支接入实机，与合并投影和位置参数表共同跑通了个体名称、写入、追问、旧值与睡眠核对。独立进程重载后读出同一个名称、新旧饮品值，数据库摘要一致。18 项集成测试通过。此轮耗时约 376 秒，峰值显存约 4.19 GB。逐题结果在 `runs/awake-block-components-v1/result.json`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_block_components.py --ceval runs/ceval-validation-617524a --output runs/awake-block-components-v1
```

## 完整 GELU 前馈的记忆回放

从第 18 层上一轮的 GELU 权重开始继续训练。两组都更新 256 步，每步抽取 64 个普通文本位置，再加入 32 个普通文本或记忆回放位置。回放取六个训练地址的 24 个查询，记录原预训练前馈在模型读取现值或旧值时的输入与输出。评测使用另外四个地址，并为每道题准备两套不同的张量。换值通过要求同一句话在两套张量下分别读对，且都没有输出另一套的预期值。

| 第 18 层配置 | C-Eval | 短对话严格正确 | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|---:|
| 预训练 SwiGLU | 18/60 | 0/16 | 7/8 | 7/8 |
| GELU 继续普通文本拟合 | 17/60 | 0/16 | 2/8 | 0/8 |
| GELU 加入记忆激活回放 | 14/60 | 1/16 | 6/8 | 4/8 |
| 回放后再对齐 72 步输出分布 | 15/60 | 1/16 | 4/8 | 1/8 |
| 回放后用两套新旧值做 72 步监督 | 14/60 | 1/16 | 6/8 | 3/8 |

记忆回放组在普通留出文本上的相对平方误差为 0.587，普通文本组为 0.577，前者却多保住四组同题换值。这次对照只覆盖固定字段、指定地址和版本，读取评分要求包含预期事实。两个输出校正实验都从同一局部回放阶段开始。显式监督组改变了目标序列、样本和损失，无法单独归因到其中一项。用独立进程重载局部回放权重，重现 6/8 固定读取、4/8 换值和 14/60 公开题。原始底座和个体检查点摘要保持一致。正式入口继续用预训练前馈。

两轮耗时约 281 秒和 207 秒，峰值显存约 4.23 GB。逐题结果和候选权重保存在 `runs/awake-gelu-replay-v1/`、`runs/awake-gelu-replay-paired-v1/` 和 `runs/awake-gelu-replay-reload-v1/`。

```powershell
.venv/Scripts/python.exe scripts/train_awake_feedforward_replay.py --start runs/awake-block-components-v1/components.pt --ceval runs/ceval-validation-617524a --output runs/awake-gelu-replay-v1
.venv/Scripts/python.exe scripts/train_awake_feedforward_replay.py --paired-targets --start runs/awake-block-components-v1/components.pt --ceval runs/ceval-validation-617524a --output runs/awake-gelu-replay-paired-v1
.venv/Scripts/python.exe scripts/train_awake_feedforward_replay.py --verify --start runs/awake-gelu-replay-paired-v1/memory_replay.pt --ceval runs/ceval-validation-617524a --output runs/awake-gelu-replay-reload-v1
```

## 用预训练投影初始化完整 GELU 前馈

将第 18 层的 SwiGLU 换成孢子的两层线性映射与 GELU。先按预训练通道在普通文本上的输出能量选取 1,536 组门控与上投影，再用两者的和、差及对应负向投影构造 6,144 个 GELU 特征。固定输入投影，通过带正则的最小二乘拟合输出。对照组从现有 GELU 回放权重提取特征，使用相同宽度、样本与输出拟合。第三组在投影初始化的基础上额外拟合两套记忆状态间的输出差。普通训练集有 3,483 个位置，两套回放状态各有 396 个位置，普通文本留出集有 369 个位置。

| 第 18 层配置 | 留出前馈相对平方误差 | C-Eval | 短对话严格正确 | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|---:|---:|
| 预训练 SwiGLU | — | 18/60 | 0/16 | 7/8 | 7/8 |
| 现有 GELU 特征重新拟合 | 0.594 | 15/60 | 1/16 | 6/8 | 5/8 |
| 预训练投影初始化 | 0.491 | 15/60 | 1/16 | 7/8 | 7/8 |
| 预训练投影初始化加记忆差分 | 0.497 | 15/60 | 1/16 | 7/8 | 7/8 |

投影初始化在受控读取和换值上达到 SwiGLU 的分数。公开题仍低三道，短对话仅通过一回合。记忆题沿用固定字段，地址和版本由评测提供。正式聊天入口继续使用预训练前馈。实验中的第 18 层仍使用预训练注意力及归一化。本轮约 208 秒，峰值显存约 4.37 GB。逐题结果及权重保存在 `runs/awake-gelu-projection-v1/`。独立进程重载投影初始化权重后，公开题、短对话和两套记忆状态的逐项结果一致。复验见 `runs/awake-gelu-projection-reload-v1/`。18 项集成测试通过。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_gelu_initialization.py --start runs/awake-gelu-replay-paired-v1/memory_replay.pt --ceval runs/ceval-validation-617524a --output runs/awake-gelu-projection-v1
.venv/Scripts/python.exe scripts/train_awake_feedforward_replay.py --verify --start runs/awake-gelu-projection-v1/projected_gelu.pt --ceval runs/ceval-validation-617524a --output runs/awake-gelu-projection-reload-v1
```

## 第 18 层完整迁入孢子解码块

用已拟合的绝对位置注意力、两处 LayerNorm 及 GELU 前馈初始化标准 `SPORALayerReplacement`。标准块在归一化前加入位置向量，并让该向量随残差传递。对照方案把同一组参数放入原有解码层接口，位置向量只进入注意力投影。整块校正使用 81 段普通文本及 24 个读入记忆的查询，训练 64 步，仅更新位置向量、归一化和注意力输出偏置。8 段留出文本上的相对残差误差从 0.506 降到 0.499。

| 第 18 层配置 | C-Eval | 短对话严格正确 | 自然结束 | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|---:|---:|
| 原预训练层 | 17/57 | 0/16 | 14/16 | 7/8 | 7/8 |
| 三个组件拼合 | 14/57 | 1/16 | 16/16 | 7/8 | 7/8 |
| 标准孢子块 | 14/57 | 1/16 | 16/16 | 7/8 | 7/8 |
| 标准块校正后 | 14/57 | 1/16 | 16/16 | 7/8 | 7/8 |

各组共用 256 词元位置上限，因长度排除 3 道 C-Eval 题。读取与换值使用受控人物和版本。校正版从磁盘重载后逐项输出一致。实机测试一度将「你好」路由为身份查询。排除纯问候触发身份读取后，「你好」进入普通生成，「你叫什么名字」仍读回「孢子」。独立个体验证了新事实写入、同人物追问、历史值、超长提示和睡眠核对。第二个进程读回姓名和新旧值，数据库摘要一致。

相关结构、清醒及集成测试合计 50 项通过。评测约 255 秒，峰值显存 4.01 GB。权重和逐题结果位于 `runs/awake-native-block-v2/`，实机输出位于 `runs/native-block-chat-v3/`。此版本保留为可选候选，普通中文仍需进一步验证。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_native_block.py --ceval runs/ceval-validation-617524a --output runs/awake-native-block-v2
```

## 第 18 层的整网输出蒸馏

从完整孢子块出发，使用原预训练第 18 层提供整网输出分布，第 6、12 层在教师和学生中均采用相同的孢子块。训练使用 83 个项目自编对话回合和 48 组受控的现值、旧值目标。每步各抽取一个对话样本和一个记忆样本，合并分布蒸馏与参考答案监督。只更新第 18 层的位置参数、两处归一化、注意力输出和前馈输出投影，共 12,198,912 个参数。

| 第 18 层 | C-Eval | 短对话严格正确 | 自然结束 | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|---:|---:|
| 起始孢子块 | 14/57 | 1/16 | 16/16 | 7/8 | 7/8 |
| 整网蒸馏 96 步 | 14/57 | 2/16 | 15/16 | 6/8 | 5/8 |

校正后读出了原先遗漏的阿宁旧值「果汁」，但丢失了阿宁当前值和小杨旧值中的「豆浆」。公开题正确数持平，尚未补回完整孢子块与预训练层之间的三题差距。此组同时使用教师分布和明确答案，无法区分两项损失各自的影响。训练及复验约 209 秒，峰值显存 3.93 GB。原模型和起始权重摘要一致，候选从磁盘重载后逐题复现。校正版保留为实验权重，现有可选聊天命令仍使用原完整块。逐题结果在 `runs/awake-native-output-v1/result.json`。

```powershell
.venv/Scripts/python.exe scripts/train_awake_native_outputs.py --start runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-validation-617524a --output runs/awake-native-output-v1
```

## 第 18 层蒸馏权重插值

对同一个完整孢子块的起始权重与 96 步蒸馏权重逐张量线性插值。每档沿用相同的 57 道 C-Eval、16 回合短对话和八组已有换值题，并新增四个训练未用地址，各查当前值和旧值。换值通过要求同一句在两套内部状态下分别答对，且两边均未输出另一套的预期值。

| 蒸馏改动保留比例 | C-Eval | 短对话严格正确 | 自然结束 | 固定状态读取 | 已有地址换值 | 新地址换值 |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 14/57 | 1/16 | 16/16 | 7/8 | 7/8 | 8/8 |
| 25% | 15/57 | 1/16 | 15/16 | 8/8 | 8/8 | 8/8 |
| 50% | 16/57 | 1/16 | 15/16 | 7/8 | 6/8 | 7/8 |
| 75% | 14/57 | 2/16 | 14/16 | 6/8 | 5/8 | 7/8 |
| 100% | 14/57 | 2/16 | 15/16 | 6/8 | 5/8 | 6/8 |

25% 候选在保持两组换值表现的同时多答对一道公开题，短对话少一次自然结束。候选选择使用了表中题目，因此仍需新题验证。校验底座和两份源权重摘要后保存候选，重载逐题输出一致。两个独立聊天进程验证了问候、名字、记忆写入、同人物追问、旧值和睡眠核对，重启后读回姓名和新旧值，数据库摘要一致。GPU 实验约 285 秒，峰值显存 4.04 GB。逐题结果与权重位于 `runs/awake-native-interpolation-v1/`，实机记录位于 `runs/native-interpolation-chat-v1/`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_native_interpolation.py --start runs/awake-native-block-v2/layer-18.pt --end runs/awake-native-output-v1/layer-18.pt --ceval runs/ceval-validation-617524a --output runs/awake-native-interpolation-v1
```

## 第 18 层插值候选的新题复验

冻结原完整孢子块和 25% 插值块。对照包括完全未改造的 DeepSeek，以及在第 6、12 层分别替换和同时替换、保留原始第 18 层的配置。语言测试均限制到 256 词元。使用此前三科选点题之外的四科验证集，共 83 道高中地理、初中历史、初中物理和中国近现代史题。另编写八组两轮短答，每轮最多 48 词元，使用真实首轮输出继续对话。受控记忆使用第 18 层蒸馏和插值选点时均未使用的六个地址，每个地址分别查询现值与旧值，再用同样的问题查询另一套已写入的值。[C-Eval 固定修订](https://huggingface.co/datasets/ceval/ceval-exam/tree/617524a00b307ff6f9933702f724131fe12ca7ce)

| 骨干 | C-Eval | 短答严格正确 | 自然结束 | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|---:|---:|
| 原始 DeepSeek | 24/83 | 0/16 | 15/16 | — | — |
| 仅第 6 层孢子块 | 26/83 | 0/16 | 14/16 | 11/12 | 9/12 |
| 仅第 12 层孢子块 | 17/83 | 0/16 | 13/16 | 9/12 | 6/12 |
| 第 6、12 层孢子块 | 17/83 | 0/16 | 16/16 | 12/12 | 11/12 |
| 加入第 18 层完整孢子块 | 17/83 | 0/16 | 15/16 | 12/12 | 11/12 |
| 第 18 层保留 25% 蒸馏改动 | 17/83 | 0/16 | 14/16 | 12/12 | 11/12 |

在仅第 6 层已替换的条件下加入第 12 层，三道题由错变对，十二道由对变错，同时受控换值从 9/12 升到 11/12。单独替换第 12 层也出现相似的语言降幅。下一轮优先校正第 12 层，并保留换值验证。相比双层版，加入第 18 层后有 23 道题改变选项，三题由错变对、另三题由对变错。插值版本只改变五道题的选项，这五题依然答错。

短答严格分会把内容正确、格式多写的答案计为错误。例如三种孢子配置都在首轮算出每盒四颗珠子，却都在第二轮复述首轮信息，漏掉剩余总数。原始版同样在该追问中答错。这里验证了指定地址下的新旧值读取，未覆盖自然语言地址解析、开放记忆或自由中文聊天。整轮没有训练，也未基于这批题重新选权重。保留 25% 候选用于实验，其选点题上的公开分数改善在本轮没有复现。

逐题输出和输入摘要位于 `runs/awake-native-holdout-v1.json`、`runs/awake-native-holdout-unmodified-v1.json` 和 `runs/awake-native-holdout-single-layers-v1.json`。复现时从固定修订下载四个科目的 `val-00000-of-00001.parquet`，按科目目录名命名为 `<subject>.parquet`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-interpolation-v1/layer-18.pt --ceval runs/ceval-native-transfer-617524a --output runs/awake-native-holdout-v1.json
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-interpolation-v1/layer-18.pt --ceval runs/ceval-native-transfer-617524a --unmodified-control --output runs/awake-native-holdout-unmodified-v1.json
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-interpolation-v1/layer-18.pt --ceval runs/ceval-native-transfer-617524a --single-layer-controls --output runs/awake-native-holdout-single-layers-v1.json
```

## 第 12 层校正与三层组合验证

从检查点中的完整第 12 层孢子块出发，固定第 6 层。83 个对话目标使用原始第 12 层的预测分布与自编参考答案，48 个受控现值和旧值目标使用双层孢子版的分布和明确值。更新位置表、两处归一化及两个输出投影，共 12,985,344 个参数。训练 96 步，重载后逐题结果一致。训练和复验约 213 秒，峰值显存 4.15 GB。

| 对照 | C-Eval 起点与校正后 | 短答严格正确 | 自然结束 | 同题换值 |
|---|---:|---:|---:|---:|
| 开发题，第 6、12 层 | 18/60 → 16/60 | 0/16 → 4/16 | 14/16 → 15/16 | 7/8 → 7/8 |
| 四科独立题，第 6、12 层 | 17/83 → 20/83 | 0/16 → 0/16 | 16/16 → 16/16 | 11/12 → 11/12 |
| 同一四科题，第 6、12、18 层 | 17/83 → 21/83 | 0/16 → 0/16 | 15/16 → 16/16 | 11/12 → 11/12 |

两组独立题的固定状态读取均保持 12/12。三层对照使用完整的原孢子第 18 层候选，所有输入共用 256 词元上限。三层评测首次运行在返回结果前退出，终端没有异常信息，故障原因未确认。重跑得到完整结果。开发集公开题退化，独立短答仍未改善，校正版权重保留为可选试验。逐题结果见 `runs/awake-layer12-output-v1/result.json`、`runs/awake-layer12-output-holdout-v1.json` 和 `runs/awake-layer12-output-combined-retry-v1.json`。

两个独立聊天进程使用校正后的第 12 层与完整第 18 层，验证了问候、名称、记忆写入、同人物追问、旧值查询和睡眠核对。重启后再次读回名称和新旧值，数据库摘要一致。实机记录位于 `runs/native-layer12-calibrated-chat-v1/`。

```powershell
.venv/Scripts/python.exe scripts/train_awake_native_outputs.py --layer 12 --ceval runs/ceval-validation-617524a --output runs/awake-layer12-output-v1
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-native-transfer-617524a --layer12-candidate runs/awake-layer12-output-v1/layer-12.pt --combine-at-18 --output runs/awake-layer12-output-combined-reproduction.json
```

## 固定第 18 层的双层对照

保留同一份完整第 18 层权重，分别还原第 12 层和第 6 层。其余评测配置沿用四科独立题，包括 256 词元上限、同一组短对话和同一对记忆状态。两组均未重新训练。

| 完整孢子块位置 | C-Eval | 短答严格正确 | 自然结束 | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|---:|---:|
| 第 6、18 层 | 24/83 | 0/16 | 14/16 | 10/12 | 8/12 |
| 第 12、18 层 | 22/83 | 0/16 | 16/16 | 11/12 | 7/12 |
| 第 6、12、18 层（第 12 层校正后） | 21/83 | 0/16 | 16/16 | 12/12 | 11/12 |

第 6、18 层在公开题上与原版 DeepSeek 的总分相同。相对未校正的三层组合，恢复第 12 层的预训练块后多答对七道公开题，少通过三项同题换值。两种双层组合的换值均低于三层版。下一轮结构改造可用第 6、18 层作为语言对照，保留校正后的三层版作为记忆对照。这组数据比较的是整层搭配，归一化、注意力和前馈各自的影响仍需逐项对照。

GPU 实验用时约 117 秒，峰值显存约 3.80 GiB，输入文件摘要通过核对。逐题输出见 `runs/awake-native-holdout-layer18-controls-v1.json`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-native-transfer-617524a --single-layer-controls --combine-at-18 --output runs/awake-native-holdout-layer18-controls-v1.json
```

## 第 12 层逐组件验证

固定第 6、18 层的完整孢子块。将校正后第 12 层的部件分别放入预训练解码层的接口，使用相同四科题、短对话和两套记忆状态。绝对位置向量在混合接口中只进入注意力投影，完整孢子块则在归一化前加入，并随残差传递。归一化分为两组，一组复制完整块权重，另一组用项目教材的 81 段文本拟合原层两处 RMSNorm 的激活，并用独立的 8 段教材检查拟合误差。

| 第 12 层配置 | C-Eval | 自然结束 | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|---:|
| 原预训练层 | 24/83 | 14/16 | 10/12 | 8/12 |
| 仅移植两处 LayerNorm | 12/83 | 0/16 | 0/12 | 0/12 |
| 仅换绝对位置注意力 | 24/83 | 14/16 | 10/12 | 6/12 |
| 仅换 GELU 前馈 | 23/83 | 16/16 | 10/12 | 6/12 |
| 三组组件放在预训练接口内 | 19/83 | 16/16 | 11/12 | 10/12 |
| 完整校正块 | 21/83 | 16/16 | 12/12 | 11/12 |
| 仅换重新拟合的两处 LayerNorm | 25/83 | 14/16 | 10/12 | 8/12 |

直接移植的两处 LayerNorm 在留出文本上的相对平方误差分别为 1.932 和 3.494。按原层输入重新拟合后降至 0.00111 和 0.00151，整网记忆得分恢复到原层水平。单独换注意力或前馈各通过 6 项换值，三组部件一起换通过 10 项，显示部件效果取决于组合。重新拟合的归一化在这批公开题多答对一道，仍比完整块少通过三项换值。各配置的严格短答均为 0/16。这批公开题已用于多轮结构比较，候选定型需另用未参与决策的新题检验。

两轮分别耗时约 382 秒和 166 秒，显存峰值约 3.91 GiB 和 4.01 GiB。输入文件摘要通过核对。逐题记录见 `runs/awake-layer12-components-holdout-v1.json` 和 `runs/awake-layer12-fitted-norms-holdout-v1.json`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-native-transfer-617524a --layer12-candidate runs/awake-layer12-output-v1/layer-12.pt --combine-at-18 --layer12-components --output runs/awake-layer12-components-holdout-v1.json
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-native-transfer-617524a --layer12-candidate runs/awake-layer12-output-v1/layer-12.pt --combine-at-18 --fit-layer12-norms --output runs/awake-layer12-fitted-norms-holdout-v1.json
```

## 重新拟合的归一化与其他组件组合

沿用同一批四科题、两套记忆及第 6、18 层。将重新拟合的两处 LayerNorm 与校正块的注意力、前馈分别组合，保持预训练层接口。

| 第 12 层配置 | C-Eval | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|
| 仅重新拟合的 LayerNorm | 25/83 | 10/12 | 8/12 |
| 加绝对位置注意力 | 25/83 | 11/12 | 7/12 |
| 加 GELU 前馈 | 23/83 | 10/12 | 6/12 |
| 两项都加 | 20/83 | 10/12 | 6/12 |
| 完整校正块 | 21/83 | 12/12 | 11/12 |

原层、仅拟合归一化及完整块的结果与上一轮重复一致。拟合后的归一化与来自联合训练块的注意力和前馈直接组合，记忆换值仍低于完整块。GPU 评测约 298 秒，峰值显存 4.01 GiB。逐题记录位于 `runs/awake-layer12-fitted-combinations-v1.json`。复现时在上一节归一化命令中加 `--combine-fitted-components` 并指定新输出路径。

## 固定第 6、18 层，再校正完整第 12 层

从已有校正候选训练 96 步。两套教师均固定完整第 6、18 层。中文目标来自原始第 12 层在 83 个项目对话上的输出，记忆目标由起始三层版为 48 个受控现值和旧值生成。沿用归一化、位置和两个输出投影的训练范围，共 12,985,344 个参数。训练与重载复验约 199 秒，峰值显存 4.00 GiB。

| 题组 | C-Eval 校正前后 | 严格短答校正前后 | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|---:|
| 开发题 | 15/57 → 18/57 | 4/16 → 3/16 | 7/8 → 7/8 | 6/8 → 6/8 |
| 四科对照 | 21/83 → 21/83 | 0/16 → 0/16 | 12/12 → 11/12 | 11/12 → 9/12 |
| 新三科 | 16/51 → 18/51 | 0/16 → 0/16 | 12/12 → 11/12 | 11/12 → 9/12 |

新三科使用同一 C-Eval 固定修订的高中生物、初中地理和高中历史。原版 DeepSeek 在相同条件下为 21/51，严格短答 0/16。新三科仅更换公开题，沿用四科对照的短答和记忆用例，所以两行的记忆分数是同一组样例的复测。这轮候选保留在实验目录，三层入口继续使用起始权重。下一轮需要在训练时约束同一问题的两种记忆状态的输出差异，再用不同地址和事实检验保持效果。

逐题记录位于 `runs/awake-layer12-three-block-replay-v1/result.json`、`runs/awake-layer12-three-block-replay-holdout-v1.json`、`runs/awake-layer12-three-block-replay-fresh-v1.json` 和同目录的 `awake-layer12-three-block-replay-fresh-deepseek-v1.json`。输入摘要通过核对。18 项结构迁移测试通过。

```powershell
.venv/Scripts/python.exe scripts/train_awake_native_outputs.py --layer 12 --start runs/awake-layer12-output-v1/layer-12.pt --fixed-layer18 runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-validation-617524a --output runs/awake-layer12-three-block-replay-v1
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-layer12-fresh-617524a --subjects high_school_biology middle_school_geography high_school_history --layer12-reference runs/awake-layer12-output-v1/layer-12.pt --layer12-candidate runs/awake-layer12-three-block-replay-v1/layer-12.pt --combine-at-18 --output runs/awake-layer12-three-block-replay-fresh-v1.json
```

## 同一问题的成对记忆训练

固定第 6、18 层，从原三层校正版训练 96 步。训练继续使用原始第 12 层的中文目标和三层起点的记忆目标。每步同时重放同一问句的两种记忆状态。在答案首次出现不同词元的位置加一个权重为 0.5 的软间隔损失，鼓励当前状态对应的答案领先另一答案。该轮同时改变了取样顺序和每步记忆样本数，现有对照无法分离成对取样与额外损失各自的作用。

| 题组 | C-Eval 起点与成对训练后 | 严格短答 | 固定状态读取 | 同题换值 |
|---|---:|---:|---:|---:|
| 开发题 | 15/57 → 15/57 | 4/16 → 4/16 | 7/8 → 8/8 | 6/8 → 8/8 |
| 已有四科与记忆组 | 21/83 → 21/83 | 0/16 → 0/16 | 12/12 → 12/12 | 11/12 → 11/12 |
| 三科与六个新人物地址 | 16/51 → 17/51 | 0/16 → 0/16 | 11/12 → 11/12 | 10/12 → 10/12 |

新人物地址未用于本轮记忆训练。两套记忆分别写入不同的现值和旧值，沿用已支持的八类饮品。受控评测直接提供地址和版本。权重从磁盘重载后复现开发题逐项结果。训练与复验约 210 秒，峰值显存约 4.00 GiB，输入文件摘要一致。

另用独立 SQLite 个体进行实机验证。名字、两次明确写入、现值、同人物追问和睡眠核对均通过。询问「齐宁上一次记录里最爱喝什么？」在本进程及重启后均返回澄清，诊断原因为 `injection_not_triggered`。用起始三层权重读取同一数据库，正确答出旧值「蜜桃乌龙茶」。两次重启读取前后数据库摘要一致。由于实机历史读取退化，启动示例继续使用起始权重。逐题结果位于 `runs/awake-layer12-paired-contrast-v1/result.json`、`runs/awake-layer12-paired-contrast-holdout-v1.json` 和 `runs/awake-layer12-paired-contrast-fresh-v1.json`。实机记录位于 `runs/native-layer12-paired-contrast-chat-v1/`。

```powershell
.venv/Scripts/python.exe scripts/train_awake_native_outputs.py --layer 12 --start runs/awake-layer12-output-v1/layer-12.pt --fixed-layer18 runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-validation-617524a --memory-contrast-weight 0.5 --output runs/awake-layer12-paired-contrast-v1
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-layer12-fresh-617524a --subjects high_school_biology middle_school_geography high_school_history --layer12-reference runs/awake-layer12-output-v1/layer-12.pt --layer12-candidate runs/awake-layer12-paired-contrast-v1/layer-12.pt --combine-at-18 --fresh-memory --output runs/awake-layer12-paired-contrast-fresh-v1.json
```

## 迁移后的历史记录读取

在同一进程切换第 12 层权重，使用同一个数据库和历史问句。起始权重在生成开头的第 5 个词元后触发读入，概率为 0.992。成对训练权重在生成开头的最高读入概率为 0.406，低于 0.65 的阈值。直接用已有的记忆解码器处理同一个归档张量时，两组都生成了正确的旧值及结束符。

已定位到人物、字段和归档版本的历史查询，在原读入时机未触发时尝试一次直接解码。生成结果必须逐词元匹配归档张量读数并正常结束。回归测试覆盖更正前读取、按时间读取、错误值、未结束输出、缺少归档和现值读取。

独立 SQLite 个体用成对训练候选通过两次写入、现值、同人物追问、旧值、睡眠核对和重启读回。旧值是「蜜桃乌龙茶」，现值是「米香玄米茶」。重启读取未改变数据库摘要。实机记录位于 `runs/native-layer12-archived-read-v1/`。按时间读取由自动测试覆盖。此次更改作用于历史记录的回答路径，成对训练权重及中文评测成绩沿用上节。

## 第 24 层完整迁移

固定第 6、12、18 层，使用原第 24 层的权重和激活初始化绝对位置注意力、LayerNorm 和 GELU 前馈。局部对齐使用中文教材及两套记忆状态回放，训练 64 步。随后用原三层模型的输出分布和参考回答对齐 96 步，并加入成对记忆训练。局部拟合在八段留出文本上将残差相对平方误差从 0.29882 降到 0.29809。

| 配置 | 三科 C-Eval | 严格短答 | 已有地址换值 | 新人物地址换值 |
|---|---:|---:|---:|---:|
| 三层起点 | 17/51 | 0/16 | 11/12 | 10/12 |
| 四层，局部对齐后 | 16/51 | 1/16 | 11/12 | 11/12 |
| 四层，整网输出对齐后 | 16/51 | 0/16 | 12/12 | 12/12 |

整网阶段更新 12,198,912 个参数。实验用时约 272 秒，显存峰值约 3.96 GiB。重载后逐项结果一致，输入文件摘要通过核对。两组换值仍属于给定人物地址和版本的受控饮品测试。三科题曾用来比较第 12 层候选，后续仍需用新题验证中文表现。

四层版通过了本机问候、名称读取、连续写入、现值、同人追问、旧值、睡眠核对和重启读回。只读重启前后数据库摘要一致。启动入口将第四块绑定到实验使用的第 12、18 层权重。实机输出在 `runs/native-layer24-fourth-block-chat-v1/`，权重和逐题记录在 `runs/awake-layer24-fourth-block-v1/`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_fourth_block.py --layer12 runs/awake-layer12-paired-contrast-v1/layer-12.pt --layer18 runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-layer12-fresh-617524a --subjects high_school_biology middle_school_geography high_school_history --output runs/awake-layer24-fourth-block-v1
```

## 第 9 层第五块实验

在已迁移的第 6、12、18、24 层固定时，将第 9 层也替换为完整孢子块。复用第 24 层的初始化、64 步局部对齐和 96 步整网对齐流程。中文评测使用固定修订中新加入的高中化学、初中政治和教育学共 69 道验证题。严格短答和两组受控记忆用例沿用前一轮。

| 配置 | C-Eval | 严格短答 | 已有地址换值 | 新人物地址换值 |
|---|---:|---:|---:|---:|
| 原版 DeepSeek | 27/69 | 0/16 | — | — |
| 四层起点 | 16/69 | 0/16 | 12/12 | 12/12 |
| 五层，局部对齐后 | 14/69 | 0/16 | 12/12 | 11/12 |
| 五层，整网对齐后及重载 | 12/69 | 0/16 | 12/12 | 12/12 |

四层与最终五层相比，高中化学从 2/19 到 3/19，教育学保持 5/29，初中政治从 9/21 到 4/21。局部对齐在八段留出文本上将残差相对平方误差从 0.70233 降到 0.68197。完整训练与重载耗时约 293 秒，峰值显存约 3.99 GiB。模型权重从文件重载后逐题结果一致，训练输入摘要通过核对。两组记忆的地址和版本由评测提供，五层版保留为实验权重，尚未进入本机聊天验收。四层版继续作为本机候选。

权重和逐题结果见 `runs/awake-layer9-fifth-block-v1/`，原版对照见 `runs/awake-layer9-fifth-block-original-v1.json`。同场对照只能归因到本轮第 9 层的整体替换，层位、初始化与对齐方案的影响仍需逐项比较。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_fourth_block.py --layer 9 --layer12 runs/awake-layer12-paired-contrast-v1/layer-12.pt --layer18 runs/awake-native-block-v2/layer-18.pt --layer24 runs/awake-layer24-fourth-block-v1/layer-24.pt --ceval runs/ceval-layer9-holdout-617524a --subjects high_school_chemistry middle_school_politics education_science --output runs/awake-layer9-fifth-block-v1
.venv/Scripts/python.exe scripts/audit_awake_native_holdout.py --native runs/awake-native-block-v2/layer-18.pt --candidate runs/awake-native-block-v2/layer-18.pt --ceval runs/ceval-layer9-holdout-617524a --subjects high_school_chemistry middle_school_politics education_science --unmodified-control --output runs/awake-layer9-fifth-block-original-v1.json
```

## 第 9 层的教师分布对照

复用同一批样本、初始化和训练步数，仅将整网阶段的人工答案损失权重从 0.25 降到 0。语言训练仍使用人工答案作为前缀，语言损失只拟合四层教师的输出分布。记忆分布、事实现值监督和成对损失保持原设定。局部阶段的残差相对平方误差与前轮一致，为 0.68197。

| 第 9 层配置 | 三科 C-Eval | 初中政治 | 严格短答 | 已有及新地址换值 |
|---|---:|---:|---:|---:|
| 四层起点 | 16/69 | 9/21 | 0/16 | 12/12、12/12 |
| 五层，局部对齐 | 14/69 | 6/21 | 0/16 | 12/12、11/12 |
| 五层，教师分布加人工答案 | 12/69 | 4/21 | 0/16 | 12/12、12/12 |
| 五层，仅教师分布 | 14/69 | 6/21 | 0/16 | 12/12、12/12 |

去掉人工答案损失后，整网阶段的换值增益保留，初中政治恢复到局部阶段水平。与四层起点相比，最终仍少答对两道公开题。本轮复用同一批公开题作开发对照，五层权重留作实验候选，尚未进入本机聊天验收。训练和重载复验约 281 秒，峰值显存约 3.99 GiB。重载后的逐项结果一致，输入摘要通过核对。逐题记录见 `runs/awake-layer9-teacher-only-v1/result.json`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_fourth_block.py --layer 9 --layer12 runs/awake-layer12-paired-contrast-v1/layer-12.pt --layer18 runs/awake-native-block-v2/layer-18.pt --layer24 runs/awake-layer24-fourth-block-v1/layer-24.pt --ceval runs/ceval-layer9-holdout-617524a --subjects high_school_chemistry middle_school_politics education_science --reference-weight 0 --output runs/awake-layer9-teacher-only-v1
```

## 第 9 层的学生前缀蒸馏

参考 GKD 对学生生成序列打分的做法，在局部对齐后让五层学生对相同的 83 个中文提示生成续写。单条最多 24 词元，且不超过原参考答案长度及位置上限，合计生成 705 个词元，39 条以结束符收尾。恢复四层教师后，缓存其在这些前缀上的下一词元分布。整网阶段保留前向 KL、96 步及原记忆损失，续写在训练前生成一次。这个实验同时改变了前缀内容和实际长度。[GKD 论文](https://arxiv.org/abs/2306.13649)

| 配置 | C-Eval | 严格短答 | 已有地址换值 | 新人物地址换值 |
|---|---:|---:|---:|---:|
| 四层起点 | 16/69 | 0/16 | 12/12 | 12/12 |
| 五层，局部对齐 | 14/69 | 0/16 | 12/12 | 11/12 |
| 五层，人工前缀，仅教师分布 | 14/69 | 0/16 | 12/12 | 12/12 |
| 五层，学生前缀，仅教师分布 | 12/69 | 0/16 | 12/12 | 11/12 |

学生前缀组少答对的两道都来自初中政治的净分变化。由于仅收集一次贪心续写，本轮结果覆盖这一固定前缀配置。完整实验约 329 秒，峰值显存约 3.99 GiB，重载复现逐题输出。权重留作实验，逐题记录见 `runs/awake-layer9-student-prefix-v1/result.json`。

```powershell
.venv/Scripts/python.exe scripts/audit_awake_fourth_block.py --layer 9 --layer12 runs/awake-layer12-paired-contrast-v1/layer-12.pt --layer18 runs/awake-native-block-v2/layer-18.pt --layer24 runs/awake-layer24-fourth-block-v1/layer-24.pt --ceval runs/ceval-layer9-holdout-617524a --subjects high_school_chemistry middle_school_politics education_science --reference-weight 0 --student-rollout-tokens 24 --output runs/awake-layer9-student-prefix-v1
```
