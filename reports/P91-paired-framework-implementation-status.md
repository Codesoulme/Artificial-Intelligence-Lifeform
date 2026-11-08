# P91 配对随机语言模型入口

本文件记录实现和运行方法，不是实验结果。本次仅执行合成 CPU 测试及只读来源摘要核对，没有启动真实训练，没有读取最终测试集，没有修改已有权重或训练脚本。

## 方案与审查

`configs/p91-paired-random-story.json` 固定两个模型种子 391109、391110，各自使用独立的数据种子；每对共享初始骨干、故事批次及标签，最多 100 次更新，评测第 0、50、100 步。目标预算为每步 8192，完整末批可超额。使用故事训练集及固定故事开发集，不读取最终测试数据、不使用 focus 或 SFT。`reports/P91-paired-random-story-review.json` 精确绑定该方案，审查依据为 P28 抽样和 P54 的故事诊断审核；它是本次 Codex 研究审查，不是新增人类盲审，也不扩大原有审核范围。

入口使用目录级操作系统互斥锁，拒绝并发运行、恢复或校验读取；锁文件保留，不应手工删除或替换。每次启动或恢复先核验完整摘要；第 50、100 步评测前以及全部完成后再次核验。每步检查已登记文件的元数据，每 10 步和指定停止点原子保存包含两臂、优化器、随机状态、轨迹和评测的联合检查点。发现元数据变化会立即停止并完整散列；元数据检查不能替代全文核验。异常最多丢失 10 步未提交工作。原个体各模块的训练模式在训练与评测期间保持不变，隔离副本用于保留探针。

每次恢复在推进前把摘要原子标为不可用；完成后发布与当前检查点摘要绑定的新报告。读取时须使用校验工具，它获取同一目录锁、核对检查点完整性，并将训练步数、目标、覆盖、轨迹及评测与检查点核对；硬件、耗时等只在摘要中的附加字段不受此内容比对保护。校验不代替来源或语义审查。断电、强制结束或发布失败可留下不可用摘要，使用相同目录及 `--resume` 恢复或重建。

## 运行与恢复

下面是将来执行的命令，本次并未运行。需要计划指定的 CUDA 设备；两对分开运行，不应并发，以免一个实验修改另一个已登记的受保护检查点：

```powershell
cd D:/SPORA
.venv/Scripts/python.exe scripts/run_p91_paired.py --plan configs/p91-paired-random-story.json --review reports/P91-paired-random-story-review.json --output runs/p91-paired-seed-391109 --seed 391109 --until 100
.venv/Scripts/python.exe scripts/run_p91_paired.py --plan configs/p91-paired-random-story.json --review reports/P91-paired-random-story-review.json --output runs/p91-paired-seed-391110 --seed 391110 --until 100
```

意外中断后，对相同目录和种子追加 `--resume`，例如：

```powershell
.venv/Scripts/python.exe scripts/run_p91_paired.py --plan configs/p91-paired-random-story.json --review reports/P91-paired-random-story-review.json --output runs/p91-paired-seed-391109 --seed 391109 --until 100 --resume
```

源码指纹已变更，旧的 10 步预检检查点不可续训；必须为后续实验选用全新的目录。联合检查点为 `checkpoint.pt`；`metadata.json` 记录方案来源、框架、设备、磁盘余量及保护文件摘要，`summary.json` 记录目标数、覆盖、批次摘要、逐文档 NLL、每臂训练耗时、评测耗时、进程分段显存峰值和校验开销。检查点是恢复依据；如果写摘要时中断，可用同一 `--resume --until` 重建摘要。读取时运行 `.venv/Scripts/python.exe scripts/read_p91_summary.py runs/p91-paired-seed-391109`，不要直接将 `summary.json` 当作已验证结果。`io-cost.jsonl` 逐次记录预检与完整核验的 SHA-256 文件读取次数、逻辑字节和耗时，以及成功提交检查点的逻辑文件字节和完整保存耗时；`summary.json` 的 `invocation_hashes` 包含本次调用至摘要生成时的哈希总量。该计数只覆盖本入口文件 SHA-256，不含训练读取、内存中的批次摘要或个体张量摘要。逻辑读取字节可能来自操作系统缓存，不等同物理磁盘读取量；写入量不含文件系统、日志和存储设备放大，保存耗时包含序列化、同步和原子替换。进程在提交后、写入成本日志前被强制终止时，该次提交成本可能缺失；没有完成的摘要不能用于估计完整调用成本。第 0 步检查点创建之前若被中断，须使用另一个新输出目录重新开始。

## 验证与限制

```powershell
.venv/Scripts/python.exe -m unittest tests.test_random_baseline tests.test_paired_experiment tests.test_paired_source_gate tests.test_p91_entry tests.test_p91_source_integration tests.test_document_stream_observation -q
```

合成测试覆盖配对初始化、标签窗口、来源审核、两个模型之间异常中断、恢复一致性、文件损坏和保护。真实源的 P28/P54、数据清单及隔离审计摘要已只读校验。

尚无完整实验的显存、时间、吞吐或效果数据。按实测 29.68 GB / 185 秒估算，不中断的 100 步单对运行约需启动、50 和 100 评测前、完成后四次完整扫描，约 118.72 GB 逻辑读取、740 秒；实际耗时还受缓存和其他来源核验影响，每次恢复及异常变化均增加扫描。元数据检查无法发现内容改动后保留或恢复原有元数据，也无法排除两次检查之间的瞬态改动；不得将其当作完整摘要。显存字段是当前进程一次调用中重置计数之后的累计峰值，包含该段评测与已有分配，不是每臂独立峰值，也不会跨恢复汇总；每步值按进程段标识分组。新实现的 CUDA 路径尚未再次实机验证；恢复要求相同设备型号和框架版本，不能承诺跨设备逐位一致。词面隔离不证明语义隔离，P28 只抽审 64 条；故事为翻译合成内容，存在质量与安全缺陷。两种种子和 100 步无法证明收敛，架构、输入信息和计算量同时不同，NLL 差异不能单独归因于记忆。现有入口不包含续写盲评或最终聊天验收。
