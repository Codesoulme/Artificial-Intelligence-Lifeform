# SPORA

探索从随机初始化的小型神经网络出发，形成可持续学习、保留自身经历并逐步扩展能力的人工生命候选系统。

当前阶段为 **P1 最小参数记忆实验**（2026-11-05），依据 v2.0 设计开始实现。首版模型为 SPORA T，969,576 个参数，随机初始化，使用真实梯度学习符号关联。

- [在线阅读完整设计](docs/SPORA-v2.md)
- [下载 Word 文档](SPORA%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E9%9D%A2%E5%90%91%E6%8C%81%E7%BB%AD%E5%AD%A6%E4%B9%A0%E4%B8%8E%E7%94%9F%E6%80%81%E6%BC%94%E5%8C%96%E7%9A%84%E6%99%BA%E8%83%BD%E4%BD%93%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1.docx)
- [首轮实践结果与已知问题](reports/P1-report.md)

核心路线是自定义结构、无预训练权重和可持久保存的神经参数记忆；巩固与局部扩容以独立验证通过为发布条件。普通推理不读取身份文档或原始经历；通过清空工作状态、冷启动和参数回退检验所学内容的来源。

设计包含最小网络规格、读写时序、从零课程、资源估算、对照实验和阶段门槛。参考文献支持局部机制，完整方案的有效性、长期成长与生命性仍是待研究问题。

当前原型能训练、保存个体、重新启动及进行只读查询。它只处理符号任务；语言、慢速巩固、环境行动和结构增长尚未完成。

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

## 文件

- `spora/model.py`：因果骨干、快速神经权重及慢速专家。
- `spora/train.py`：短展开元训练；`spora/evaluate.py`：独立随机世界与重启审计。
- `spora/checkpoint.py`：原子保存、恢复与学习状态；`spora/cli.py`：符号交互入口。
- [实施边界](docs/P1-implementation.md)与[预设审计计划](configs/p1-audit-plan.json)。
- `runs/` 保存本地指标，`checkpoints/` 保存权重；虚拟环境、经历记录与张量不进入版本控制。

重新配置环境时使用 Python 3.11 和 `requirements.txt`；完整已安装版本见 `requirements-lock.txt`。不需要下载任何预训练模型。
