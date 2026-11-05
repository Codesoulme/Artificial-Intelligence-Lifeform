# SPORA

探索从随机初始化的小型神经网络出发，形成可持续学习、保留自身经历并逐步扩展能力的人工生命候选系统。

当前完成 **P11 真实时长与有限环境预测**（2026-11-05），中文训练继续暂停。新增随机初始化的小型预测模块、单调时钟与持久工作状态，通过旧记忆、事务恢复和历史睡眠兼容检查。原有个体继续保留，新副本继承睡眠候选的72次符号观察。

- [中文训练前交付报告](reports/prelanguage-report.md)与[完整结果，包括失败版本](reports/prelanguage-results.json)
- [P8 读取优化实测](reports/P8-read-report.md)与[睡眠、存储研究和实施顺序](docs/memory-read-sleep-storage.md)
- [P9/P10 交付报告](reports/P9-P10-report.md)、[完整数值与对照](reports/P9-P10-results.json)及[实现与命令](docs/durable-sleep-implementation.md)
- [P11 时间与环境预测报告](reports/P11-report.md)、[完整结果](reports/P11-results.json)及[设计、边界与命令](docs/time-world-implementation.md)
- [后续待办](docs/TODO.md)：完整睡眠想象、中文语言能力训练；两项均尚未启动。
- [TencentDB Agent Memory 评审](docs/tencentdb-memory-review.md)：区分内部机制、后续学习内容与外部工具能力。
- [在线阅读总体设计](docs/SPORA-v2.md)及[P5–P7 结构补充](docs/P5-P7-design.md)；下列 Word 文件保留为设计基线。
- [下载 Word 文档](SPORA%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E9%9D%A2%E5%90%91%E6%8C%81%E7%BB%AD%E5%AD%A6%E4%B9%A0%E4%B8%8E%E7%94%9F%E6%80%81%E6%BC%94%E5%8C%96%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1.docx)
- [首轮实践结果与已知问题](reports/P1-report.md)
- [第二轮实测结果、失败与边界](reports/P2-report.md)及[实验协议](docs/P2-protocol.md)

核心路线是自定义结构、无预训练权重和可持久保存的神经参数记忆；巩固与局部扩容以独立验证通过为发布条件。符号回忆不读取身份文档或原始经历；通过清空工作状态、冷启动和参数回退检验所学内容的来源。时间模块另存最近的数值传感状态，明确属于工作状态，不冒充长期参数记忆。

设计包含最小网络规格、读写时序、从零课程、资源估算、对照实验和阶段门槛。参考文献支持局部机制，完整方案的有效性、长期成长与生命性仍是待研究问题。

当前原型能训练、保存个体、重新启动及进行只读查询。有限符号实验已验证历史保持、未知拒答、分段成长及封存历史转存；新增简单环境的2秒内预测。语言、开放实体理解、自主行动和睡眠想象尚未形成。P3 的慢速专家机制仍未合入；早期谱系也没有自动迁入最新候选。

当前仓库不包含权重文件。在本工作目录内模型已保存在 `checkpoints/`；在另一台电脑克隆源码后，需要从随机初始化运行训练，再通过 `create` 建立个体。

## 最新：真实时长与简单环境预测

新增4,932个参数，学习信号间隔及给定动作下的位置、速度、温差变化。三个种子均通过固定课程门槛；相同总时长、不同调用次数的输出一致。重启间隔只估计，不补写观察；超出2秒窗口要求新输入。此阶段不证明主观时间体验，也没有完成世界模型驱动的做梦。

本地 `checkpoints/individual-time-candidate.sqlite` 是新的事务个体，`.pt` 是发布时快照。原有72次符号经历保持，初始时间工作状态为空。实际等待演示在独立副本运行，退出后没有后台服务。操作方式见[时间模块说明](docs/time-world-implementation.md)。

85项回归和七处进程中断检查通过；旧张量、回忆输出、睡眠兼容性均通过核对。周期头误差约0.79%–0.89%，仍逊于简单取平均的0.67%。完整成本和一次审计逻辑修正均记录在[P11报告](reports/P11-report.md)。

## P9/P10：增量保存与有限历史睡眠

SQLite WAL 事务只提交变化的神经张量、清单和游标，检查旧版本写者；普通观察和睡眠替换各通过七个位置的强制进程中断测试。恢复结果、下一次写入、备份和整理均通过。它保存模型状态，不提供符号答案表。

睡眠把写满的历史段蒸馏到随机初始化的 rank16 神经因子，并精确打包辅助矩阵；骨干和当前通道保持不变。三个基础种子、各两个环境、每条链三次转存全部通过，随后共48次可信纠正也通过。只检查过有限符号环境，不代表具备语义抽象或世界模型想象。

每个封存段状态减少56.94%；三段已封存的完整便携文件仅缩小约4.5%，且读取压缩区域需要重建计算。SVD 对照也达到相同有限任务正确率，因此本轮没有证明训练优于解析压缩。详见报告中的成本、对照与未解决项。

本地 checkpoints/individual-durable.sqlite 保留原三次经历；checkpoints/individual-sleep-candidate.sqlite 是从它继续的独立分支，追加69次程序生成观察并完成一次睡眠，已有72次经历。对应便携文件为 checkpoints/individual-sleep-candidate.pt。

~~~powershell
.\.venv\Scripts\python.exe -m spora.durable_cli --individual checkpoints/individual-sleep-candidate.sqlite info
.\.venv\Scripts\python.exe -m spora.durable_cli --individual checkpoints/individual-sleep-candidate.sqlite ask 12
.\.venv\Scripts\python.exe -m spora.durable_cli --individual checkpoints/individual-sleep-candidate.sqlite ask 12 --time 0
~~~

后两条回答当前7、最初观察5。备份使用 durable_cli backup；sleep 只接收已满的历史段并在验收后提交。不要直接复制一个正在写入的 SQLite 主文件而忽略日志。

## P8：减少重复读取，保留完整回退

字节接口默认合并数值相同的矩阵读取，保留每个段的位置权重。可选 bounded 路径只有在遗漏影响的上界足够小时才跳过其他段；目前接口尚未学习语义，通常需要完整读取，所以没有默认启用近似筛选。

48 个已有个体在 CPU/GPU 的输出保持检查通过。额外测试 1/4/8/16 段分配、单位置与批量输入，既报告加速也报告变慢的条件；16 段包含新增空段，不是 1,024 次回忆验收。完整计时、数值偏差、运行内存和未完成项见 P8 报告。

P8 阶段时，P3 只验证了旧结构上的有限转存；后续事务保存与新版历史巩固已由上述 P9/P10 实施。世界模型想象仍未实现，中文训练也没有启动。

## P5–P7：中文训练前停止点

本轮完成七个有限训练实验，共 4,200 个外层更新、380,800 次事件呈现。先发现 P5 在 64 个不同实体下的当前值与未知判断失败，再以 P5b 新开发流修订，并用新的审计流验收。

- 三个随机种子在 8/16/32/64 次事件审计中的当前值、历史和早期历史均为 100%；64 次事件错误组合拒答为 99.88%–99.90%。这些是有限符号任务结果，不是自然语言成绩。
- 连续成长覆盖三个种子、两种实体分布、各八个世界；每次连续经历 256 个事件，逐段保存与恢复，所有预定指标通过。增加容量需要增加内部矩阵和优化器状态。
- 另测同键错误时刻，拒答率为 98.44%–98.83%；并非所有未知问题都能正确识别。
- 56 项回归通过，三份旧个体文件摘要不变；新候选从通过验收的 P5b 谱系精确迁入增长格式，已有经历无需重放。
- 新字节接口只完成随机输入的机制检查，实际语言训练 token 数为零；首次中文数据方案见[数据准备文档](docs/chinese-data-plan.md)。

本地候选位于 checkpoints/individual-prelanguage.pt，共 1,042,546 个模型参数、11,520 个固定地址标量和 145,800 个活动状态标量；四段已分配容量为 256 次经历。未训练接口位于 runs/p7-preparation/untrained-interface.pt，另含 190,037 个可训练参数。它共享并冻结候选的因果骨干，只读访问记忆；中文与实体、关系和时间的对齐仍未学习。

只读查看这个候选：

~~~powershell
.\.venv\Scripts\python.exe -m spora.growing_cli --individual checkpoints/individual-prelanguage.pt info
.\.venv\Scripts\python.exe -m spora.growing_cli --individual checkpoints/individual-prelanguage.pt ask 12
.\.venv\Scripts\python.exe -m spora.growing_cli --individual checkpoints/individual-prelanguage.pt ask 12 --time 0
.\.venv\Scripts\python.exe -m spora.growing_cli --individual checkpoints/individual-prelanguage.pt ask 12 --time 2
~~~

后三条依次回答当前 9、早期观察 5、不可信观察 3。候选只携带三次演示经历；256 次连续成长是在独立审计状态上验证，并非声称交付个体已经经历了这些事件。全部成绩、消融、参数成本和未解决问题见报告。

本轮停在实际中文语料训练之前。spora.train_language 的启动参数不是语言能力证明；准备接口不会触发训练。实现与验收顺序见[阶段路线](docs/pre-language-roadmap.md)，复现实验入口为 scripts/batch_p5.py 和 scripts/batch_p5b.py，均应使用独立的本地实验目录。

## 本地运行

工作目录：`D:\SPORA`。现有独立环境位于 `.venv`，已使用 GTX 1060 Max-Q 验证前向、反向与二阶梯度。

```powershell
Set-Location 'D:\SPORA'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m spora.cli info
.\.venv\Scripts\python.exe -m spora.cli learn 12 5
.\.venv\Scripts\python.exe -m spora.cli ask 12
```

`learn 12 5` 表示亲历一次“符号 12 对应结果 5”。`ask 12` 在新进程中读取参数，不读取前次命令或事件文本。当前键范围 0–63，结果范围 0–15；默认操作同一份 `checkpoints/individual.pt`。模型容量有限，不承诺无限记忆。

本节默认 P1 的已知限制：初次学新关联与纠正旧关联并不等价，其一次纠错成功率较低，不能用于要求可靠覆盖旧事实的任务；增加经历数量会干扰早期记忆。默认个体已经历一次演示映射 `12 → 5`。最新候选须使用前述 growing_cli 路径查询。

新建分支个体，或运行新的有限训练：

```powershell
.\.venv\Scripts\python.exe -m spora.cli --individual checkpoints/branch.pt create --foundation checkpoints/p1-seed7.pt
.\.venv\Scripts\python.exe -m spora.train --steps 1000 --seed 31 --device cuda --run-dir runs/p1-seed31 --checkpoint checkpoints/p1-seed31.pt
```

已存在的训练输出不会被静默覆盖。断点继续用 `--resume`，并保持原先 seed、batch-size 与 pairs；`--steps` 表示额外训练步数。

## P2 独立候选

默认交互仍使用 P1。新候选位于 checkpoints/individual-p2-candidate.pt，已经历 12→5 后再纠正为 12→9。它属于新谱系，没有自动继承或替换原个体。

~~~powershell
.\.venv\Scripts\python.exe -m spora.cli --individual checkpoints/individual-p2-candidate.pt ask 12
~~~

learn 可选 --policy one、two 或 adaptive-max4，以及 --threshold。一次外部经历仅递增一次计数，内部最多四次计算后原子保存。首次缺省为一步，此后沿用个体保存的策略；查询不会修改文件。

P2 入口包括 spora/train_p2.py、compare_p2.py、stress_p2.py、protection_p2.py 和 consolidate_p2.py；scripts/report_p2.py 汇总证据。仍只支持有限符号任务，详细失败和限制见报告。

## P3 巩固修订实验

本轮重点检查转存后能否继续纠正，以及其他已学事实是否保留。新增残差写入是独立候选选项，原模型文件加载时仍采用原规则；不会自动改变默认个体。

- [P3 结果及失败边界](reports/P3-report.md)
- [固定实验协议](docs/P3-protocol.md)与[计划配置](configs/p3-plan.json)
- spora/consolidate_p3.py 执行有界开发与评估；scripts/audit_p3.py 锁定开发选择后执行三组评估。
- 精确矩阵搬移仅作不压缩对照，不等于增加容量或完成抽象学习。

本地 P3 候选为 checkpoints/individual-p3-candidate.pt，携带一次独立实验的 24 条已巩固事实。它从固定 seed7/world0 导出，没有替换原个体。测试用清单在 reports/P3-candidate.json；模型推理不读取该文件。

~~~powershell
.\.venv\Scripts\python.exe -m spora.cli --individual checkpoints/individual-p3-candidate.pt ask 56
~~~

当前此查询应回答 4。后续主动使用 learn 会改变该候选；残差写入按其检查点配置启用，无需每次传入策略。

## P4 时间记忆实验

新结构支持 current(k) 与 event(k,t)：前者回答最后一次可信赋值，后者回答在指定时刻实际收到的内容。历史包含不可信观察；记得某句话不等于认定它为真。可信度仍由外部环境提供。

- 实验方案见 [P4 协议](docs/P4-protocol.md)与 [固定配置](configs/p4-plan.json)；完成结果见 [P4 报告](reports/P4-report.md)。
- 两种结构均重新随机训练：split 分开当前及历史神经矩阵，shared 共用矩阵。两者近似匹配活动快参数，未匹配总计算量。
- 普通查询只输入键、模式和时间；经历在个体神经参数中持久保存，不读取事件表。当前时间词表只有 0–63，容量满时明确拒绝继续写入。
- 六个研究底座与审计用状态保存在本地 checkpoints/、runs/，不随源码上传。未过门槛的底座不能视为新的发布个体。

本轮审计已完成：split 在 8、16、32 事件上的历史准确率分别为 98.89%、96.42%、84.99%，相应当前值均为 100%；64 事件压力下历史降至 60.21%。以上为三种子的有限任务汇总，单个种子更早也可能失败。六个模型均未通过全部预定门槛，因此未导出 P4 发布候选。38 项回归检查通过，原有三份个体校验值不变。

复现实验（使用新的运行目录；已有结果不会覆盖）：

~~~powershell
.\.venv\Scripts\python.exe -m spora.train_p4 --kind split --seed 31 --run-dir runs/p4-split-seed31 --checkpoint checkpoints/p4-split-seed31.pt
.\.venv\Scripts\python.exe -m spora.audit_p4 --checkpoint checkpoints/p4-split-seed31.pt --output-dir runs/p4-split-seed31-audit
~~~

只在独立研究路径试用交互入口；不会自动升级现有默认个体：

~~~powershell
.\.venv\Scripts\python.exe -m spora.temporal_cli --individual checkpoints/my-p4-experiment.pt create --foundation checkpoints/p4-split-seed7.pt
.\.venv\Scripts\python.exe -m spora.temporal_cli --individual checkpoints/my-p4-experiment.pt learn 12 5
.\.venv\Scripts\python.exe -m spora.temporal_cli --individual checkpoints/my-p4-experiment.pt learn 12 9
.\.venv\Scripts\python.exe -m spora.temporal_cli --individual checkpoints/my-p4-experiment.pt learn 12 3 --trusted 0
.\.venv\Scripts\python.exe -m spora.temporal_cli --individual checkpoints/my-p4-experiment.pt ask 12
.\.venv\Scripts\python.exe -m spora.temporal_cli --individual checkpoints/my-p4-experiment.pt ask 12 --time 0
~~~

上述目标分别为当前 9、t0 观察为 5；具体回答属于模型预测，不能保证正确。无 --time 为当前查询；提供 --time 则询问过去观察。每次 learn 推进一次事件计数；ask 不保存。

## 文件索引

- spora/temporal.py：当前与历史分离、RLS 写入及未知判别；spora/growing.py：有限分段成长；spora/growing_cli.py：新个体交互入口。
- spora/language_bridge.py：未训练字节接口；spora/text_data.py 与 spora/train_language.py：待启用的数据检查和训练入口。
- scripts/report_prelanguage.py：汇总各轮证据并检查停止条件。
- `spora/model.py`：因果骨干、快速神经权重及慢速专家。
- `spora/train.py`：短展开元训练；`spora/evaluate.py`：独立随机世界与重启审计。
- `spora/checkpoint.py`：原子保存、恢复与学习状态；`spora/cli.py`：符号交互入口。
- [实施边界](docs/P1-implementation.md)与[预设审计计划](configs/p1-audit-plan.json)。
- `runs/` 保存本地指标，`checkpoints/` 保存权重；虚拟环境、经历记录与张量不进入版本控制。

重新配置环境时使用 Python 3.11 和 `requirements.txt`；完整已安装版本见 `requirements-lock.txt`。不需要下载任何预训练模型。
