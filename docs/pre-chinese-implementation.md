# P14–P16：中文训练前的内部能力与工程准备

2026-11-05。本轮只执行有限符号、数值反馈和随机字节课程；没有用中文文本更新参数。总体结论与完整数据见 [交付报告](../reports/P14-P16-report.md)、[机器可读结果](../reports/P14-P16-results.json)。

## 快慢层接口

`spora.semantic_memory` 从当前神经矩阵读取事实，并以内部证据矩阵筛选具有已记录来源、质量至少0.8的条目。至少16条才能提出候选。旧记录没有来源时不补造来源，也不自动拿来归纳。

有限任务的结构先验是：64个符号可以分解为六个二值属性，16个取值可以分解为四个输出位。慢层是6→4的线性神经头，共28个参数。800步有界训练只接触神经记忆读出的标签，训练样本不持久保存为答案表。已接受的当前事实全部复核；任何一个输出不一致，或正确位的分数低于0.9，候选就不发布。

这属于给定属性空间内的有限规律归纳，不是自行发现自然语言概念，也没有解决任意多任务的语义压缩。二值分解、线性结构和置信门槛都是工程先验；输出分数不是经过开放世界校准的正确概率。

查询先返回已观察事实，只有未知项才允许使用通过门槛的慢层。结果必须携带 `observed`、`inferred` 或 `unknown` 分类。发生新的符号观察后，旧归纳暂时失效，复核成功才能再次用于未知项。这个保守策略也会被重复消息触发；它以额外重训开销换取明确的版本关系。

`consolidate_semantic_store` 在独立候选上训练，接受后通过原有事务存储发布；拒绝不会改动原个体。它不删除历史，不把推论写成事实，也不冒充完整的睡眠想象。P10的历史压缩和这里的概念抽取是两个操作，需要分别通过各自验收。

## 动作反馈接口

`spora.skills.FeedbackPolicy` 在有限二维环境中学习五个动作的价值。输入是目标与位置的差及其二次项，这是显式提供的特征；30个线性参数从随机值开始训练。`learn_feedback` 接受位置、目标、实际动作、结果位置和标量反馈，更新被执行动作的预测。结果位置检查维度和数值，学习目标来自环境给出的反馈；没有独立验证奖励是否合理。

课程执行随机探索，共1500批、每批128条反馈。训练脚本包含模拟环境和奖励公式，推理模块只使用所学参数。训练排除两个同号角落区域；审计在这些区域开始，并平移绝对目标位置，检验有限迁移。

这不是通用强化学习智能体。离散步长、相对坐标和二次特征使该奖励函数特别容易表示；解析规则对照与所学策略同样有效。技能没有连接P11的力／温度环境，也没有接受中文指令。两者之间的任务映射要在后续单独建立。模块保存权重和更新次数；新的反馈训练会显式建立优化器，不声称支持跨任务无遗忘或技能优化器的精确续训。

## 新个体与谱系

本地 `checkpoints/individual-prechinese-v3.pt` 和同名 `.sqlite` 继承P12版本的72次观察。全部旧张量保持相等，新增能力张量共256字节；便携文件增加3058字节，事务文件本轮未增加文件大小，因为原有页面可容纳新增记录。

加入的二维策略已经学习过数值反馈。原个体没有符合门槛的来源标注概念课程，因此慢层保留随机初始权重和未启用标记，查询会拒绝使用它推断。实验个体学到的属性规则没有移植进来。`runs/p16-runtime-audit-v2/semantic-demonstration.pt` 是独立的合成课程后代，验证实际发布、冷启动与历史共存。

所有扩展跟随同一神经快照和事务保存。语言接口是与父文件摘要绑定的独立新增权重；调用者不能通过接口载入路径替换个体骨干、计时模块、来源控制或新能力层。

## 使用方式

先复制事务个体到新的试验路径，避免改变已审计的交付文件。以下命令中的 `checkpoints/my-trial.sqlite` 指向用户选择的副本。

```powershell
.\.venv\Scripts\python.exe -m spora.capabilities_cli --individual checkpoints/my-trial.sqlite infer 12
.\.venv\Scripts\python.exe -m spora.capabilities_cli --individual checkpoints/my-trial.sqlite act --position 0 0 --goal 0.8 0.4
.\.venv\Scripts\python.exe -m spora.capabilities_cli --individual checkpoints/my-trial.sqlite consolidate
```

前两个只读取，动作建议不会执行环境操作。最后一个尝试有界巩固，无足够来源事实时直接拒绝。已有 `spora.internal_cli` 仍负责来源标注的观察写入。

## 中文入口的变化

- 新增严格接口载入：父文件摘要、完整字段集合、形状、精度与有限值必须匹配；不接受额外的个体张量。
- 全个体冻结，只优化190037个语言接口参数。保存和恢复训练步数、Adam状态、抽样状态及计算随机状态；CPU连续训练与断点继续逐值一致。GPU完成随机字节更新与资源检查，但没有宣称GPU断点逐位一致。
- 每10步原子保存一次，恢复时使用新输出目录，额外步数加上历史步数不得超过10000。断点前尚未保存的至多9步会丢失。文件替换失败保留上次完整检查点；没有做真实断电测试。
- 训练结束复核父文件和内存中全部个体张量摘要。数据清单、设备、种子、批次与长度变化时禁止直接续训。
- 逐文档分区，检查来源说明、使用范围、摘要、UTF-8、空文本、规范化重复，以及跨分区的13字符片段重叠。至少24个共同片段且较短文档的片段覆盖率达到80%时拒绝。此过滤不识别语义改写，也不保证检出很短或局部比例低的重复。
- 首轮清单限制4096个文档、总计4MB，以避免未审查地扩大课程。没有通过此限制来推断内容或许可已经合格；第三方数据仍需单独审阅。
- `spora.language_eval` 提供按实际目标数加权的损失、每字节或结束符的比特数、固定提示续写和原始UTF-8有效性。解码异常会如实报告，显示替代字符不算有效中文。

## 首次试验材料

本项目编写了36篇合成短文，按24／6／6分为训练、开发、封存测试。内容是日常描述、简单数量关系、条件和观察，不包含个体身份设定。源文本位于 `data/chinese-pilot/source.json`；本地清单在 `runs/chinese-pilot-v1/manifest.json`。训练部分只有3531个UTF-8字节，反复学习极易过拟合，只用于小规模工程验证。

建议首轮预算已固定在 [配置](../configs/chinese-pilot-plan.json)。以下是准备好的未来执行命令，**本轮未运行**：

```powershell
.\.venv\Scripts\python.exe -m spora.train_language --individual checkpoints/individual-prechinese-v3.pt --interface runs/p16-language-ready-v2/untrained-interface.pt --manifest runs/chinese-pilot-v1/manifest.json --output runs/chinese-pilot-train1 --start-language-training --steps 100 --batch 4 --length 128 --seed 1910000 --device cuda
```

实际启动还必须追加 `--review <approved-review.json>`。该审查文件需要绑定精确 manifest 摘要、父个体摘要、步数上限和独立质量门禁；当前没有这样的批准记录，因此上面的命令仍只是历史计划示例，不能直接启动训练。

P152 增加了可选的 `--context-layers 2` 路径。它生成 v2 接口并在冻结个体前增加可学习的因果上下文混合器；默认值仍为 0，以保持旧接口兼容。v2 只完成了真实个体初始化烟测，尚未进行中文更新。

先测未训练接口的开发损失，再训练，最后只做一次封存测试；不能看测试表现反复选模型。首次评估可以使用以下入口：

```powershell
.\.venv\Scripts\python.exe -m spora.language_eval --individual checkpoints/individual-prechinese-v3.pt --interface runs/chinese-pilot-train1/checkpoint.pt --manifest runs/chinese-pilot-v1/manifest.json --split test --output runs/chinese-pilot-test1.json --prompt "清晨，" --prompt "桌上有" --prompt "如果下雨，"
```

若需要续训，用 `--resume 原运行/checkpoint.pt` 替换 `--interface`，其余数据、设备与批次参数保持一致，并提供新的输出目录。

语言接口目前读取当前和历史矩阵，没有学会词语到事实地址的映射，也没有访问新慢层或动作模块的语言通道。纯字节预测不能被解释为“已经能用中文讲述自己的经历”。第一次工程试验后，再根据结果设计更大的许可清晰语料和事实对齐课程。
