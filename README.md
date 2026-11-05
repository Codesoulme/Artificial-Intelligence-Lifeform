# SPORA

探索从随机初始化的小型神经网络出发，形成可持续学习、保留自身经历并逐步扩展能力的人工生命候选系统。

当前保留 P1 默认个体、P2 与 P3 候选，新增 **P4 当前判断与历史经历分离的研究实现**（2026-11-05）。P4 另起随机权重谱系，不自动迁移旧个体；所有基础谱系没有采用预训练语言模型。

- [在线阅读完整设计](docs/SPORA-v2.md)
- [下载 Word 文档](SPORA%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E9%9D%A2%E5%90%91%E6%8C%81%E7%BB%AD%E5%AD%A6%E4%B9%A0%E4%B8%8E%E7%94%9F%E6%80%81%E6%BC%94%E5%8C%96%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1.docx)
- [首轮实践结果与已知问题](reports/P1-report.md)
- [第二轮实测结果、失败与边界](reports/P2-report.md)及[实验协议](docs/P2-protocol.md)

核心路线是自定义结构、无预训练权重和可持久保存的神经参数记忆；巩固与局部扩容以独立验证通过为发布条件。普通推理不读取身份文档或原始经历；通过清空工作状态、冷启动和参数回退检验所学内容的来源。

设计包含最小网络规格、读写时序、从零课程、资源估算、对照实验和阶段门槛。参考文献支持局部机制，完整方案的有效性、长期成长与生命性仍是待研究问题。

当前原型能训练、保存个体、重新启动及进行只读查询。P3 慢速巩固和后续修订已通过有限符号检查；P4 增加有限时刻查询与 unknown 类别，高负荷历史仍可能遗忘。语言、环境行动和结构增长尚未完成，两个阶段的机制尚未合并。

当前仓库不包含权重文件。在本工作目录内模型已保存在 `checkpoints/`；在另一台电脑克隆源码后，需要从随机初始化运行训练，再通过 `create` 建立个体。

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

已知限制：初次学新关联与纠正旧关联并不等价。当前模型的一次纠错成功率较低，不能用于要求可靠覆盖旧事实的任务；增加经历数量会干扰早期记忆。默认个体已经历一次演示映射 `12 → 5`。

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

- `spora/model.py`：因果骨干、快速神经权重及慢速专家。
- `spora/train.py`：短展开元训练；`spora/evaluate.py`：独立随机世界与重启审计。
- `spora/checkpoint.py`：原子保存、恢复与学习状态；`spora/cli.py`：符号交互入口。
- [实施边界](docs/P1-implementation.md)与[预设审计计划](configs/p1-audit-plan.json)。
- `runs/` 保存本地指标，`checkpoints/` 保存权重；虚拟环境、经历记录与张量不进入版本控制。

重新配置环境时使用 Python 3.11 和 `requirements.txt`；完整已安装版本见 `requirements-lock.txt`。不需要下载任何预训练模型。
