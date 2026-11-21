# 第 18 层的查询、键和值位置迁移

在第 26 层门控检查点上，复用第 18 层训练好的绝对位置参数。选第 0 键值头组做两种切换。第一种只换它的六个查询头与一个键头，值保持父模型计算。第二种只换值投影，查询和键完整保留原有旋转位置。其余头组和注意力输出投影均沿用父模型。每个配置共享原有的键值头数量和生成状态管理。

两种候选各跑 60 道 C-Eval、24 轮开发对话、32 条受控当前／历史记忆回答。在八条普通首轮提示上比较末位置全词表 KL，并验证一条普通生成和一条记忆生成在开启及关闭键值复用时的词元一致性。先前的整组切换同时改变三项投影，在相同题目上为 `23/60`。

## 结果

| 第 0 组的变更 | 平均词表 KL | C-Eval | 与父模型逐词元一致的对话 | 受控记忆 |
|---|---:|---:|---:|---:|
| 只换查询与键 | 0.001815 | 23/60 | 12/24 | 32/32 |
| 只换值 | 0.000468 | 24/60 | 24/24 | 32/32 |
| 查询、键和值一起换 | 0.002198 | 23/60 | 12/24 | 32/32 |

父模型为 `24/60`。只换值时 60 道题逐题保持原选项。两种新的单部件候选均通过两条生成复用检查。这批题上的语言退化主要随查询和键一起出现。值路径保留为独立研究组件，包中只有绝对位置表和一个值头的权重、偏置，推理只增加一次窄值投影。

## 独立进程重载

重新加载父检查点和 2,754,516 字节的值路径组件后，原有 60 道题和额外 83 道题的选项都与父模型逐题一致，分别答对 `24/60` 和 `23/83`。24 轮对话逐词元一致，18 轮自然结束。32 条受控记忆回答逐项复现，四个问句的八值替换均通过，两条生成复用检查通过。该结果支持把值路径加入后续实验骨干。对话指标仍处于父模型水平，自动寻址与自由聊天需要各自的评测。

组件依赖已验证的第 26 层门控父检查点，加载时检查父文件和组件摘要。组件 SHA-256 为 `a1042f7d7ec4973bdad112e252bf1139a6b5bbcdd8e4cf24266ff241f6d0dcc2`。对照和重载分别耗时约 319 秒和 212 秒，峰值显存分配均约 4.95 GiB。CPU 侧另有 44 项相关测试通过。详细记录在 `runs/spora-position-components-v1/result.json` 与 `runs/spora-packed-value-position-v1/verified.json`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe scripts/audit_spora_split_position_attention.py --checkpoint runs/spora-language-migration-layer26-v1/candidate.pt --attention runs/spora-grouped-absolute-attention-v1/attention.pt --baseline runs/spora-language-migration-layer26-v1/migration.json --reload-report runs/spora-language-migration-layer26-v1/reload.json --ceval runs/ceval-validation-617524a --components --group 0 --output runs/spora-position-components-v1/result.json
.\.venv\Scripts\python.exe scripts/verify_spora_value_position_component.py pack --checkpoint runs/spora-language-migration-layer26-v1/candidate.pt --audit runs/spora-position-components-v1/result.json --attention runs/spora-grouped-absolute-attention-v1/attention.pt --output runs/spora-packed-value-position-v1
.\.venv\Scripts\python.exe scripts/verify_spora_value_position_component.py verify --checkpoint runs/spora-language-migration-layer26-v1/candidate.pt --audit runs/spora-position-components-v1/result.json --baseline runs/spora-language-migration-layer26-v1/migration.json --reload-report runs/spora-language-migration-layer26-v1/reload.json --ceval runs/ceval-validation-617524a --output runs/spora-packed-value-position-v1
```

RoVE 为值路径单独引入相对位置信号，并在小规模模型的长距检索和聚合任务上测试了收益。本轮测试用的是孢子的可学习绝对位置，只借用分别考察分数路径和值路径的思路。 [RoVE 预印本](https://arxiv.org/abs/2606.11275)
