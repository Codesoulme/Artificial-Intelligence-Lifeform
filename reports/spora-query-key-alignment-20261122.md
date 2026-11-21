# 第二头组的位置迁移与局部对齐

从已验证的第 18 层单组检查点出发，固定语言骨干、第一头组的投影、位置表和个体。第二头组依次试最低频 12、8、4 对坐标。普通中文开发对话增加无蛋早餐的第三轮复验。该复验只检查食材列表是否重新加入鸡蛋，父模型在这轮也没有满足一句话的格式要求。

| 第二头组 | 三科 C-Eval | 原始选项一致 | 8 个首轮提示的平均词表 KL | 食材复验 |
|---|---:|---:|---:|---|
| 12 对 | 25/60 | 60/60 | 0.000792 | 加入鸡蛋 |
| 8 对 | 25/60 | 60/60 | 0.000671 | 加入鸡蛋 |
| 4 对 | 25/60 | 60/60 | 0.000608 | 加入鸡蛋 |
| 8 对，局部对齐 96 步 | 25/60 | 60/60 | 0.000218 | 加入鸡蛋 |

局部对齐从原有绝对位置投影初始化，只更新第二头组的查询与键权重、偏置。在 32 段普通中文和 20 条带记忆输入上拟合父模型的注意力分布与该头组的值加权输出。12 条局部留出样本上，平均注意力 KL 从 `0.004882` 降至 `0.001974`，相对输出均方误差从 `0.003297` 降至 `0.001064`。量化到实际半精度后测得这些结果。局部拟合更接近父模型，但仍未保住多轮中的饮食约束。四个候选都停在该复验，尚未运行独立新对话和 32 条受控记忆换值。逐题证据在 `runs/spora-query-key-expansion-v2/result.json` 和 `runs/spora-query-key-local-alignment-v1/result.json`。两轮分别约 213 秒和 146 秒，峰值 CUDA 分配均约 4.94 GiB。第一次启动在训练前发现原定语料切片只有 9 条，重跑改用语料末尾的 40 条。

阶段式蒸馏提供了下一步的具体对照。在同一候选上固定局部训练，再用普通多轮对话做整网预测对齐，检查新增句式和内部记忆换值是否一起保留。另可按实际注意力贡献选坐标，验证低频排序对第二组造成的限制。相关论文的迁移目标与此处不同，本轮只使用了局部对齐。 [MOHAWK](https://arxiv.org/abs/2408.10189) · [MLAfication](https://arxiv.org/abs/2610.17202)

## 检查点与增量状态

标准检查点支持同层保存多组查询与键的坐标和投影，继续读取已有的单组格式。增量状态绑定检查覆盖各组编号、坐标和权重。小型双头组测试验证了保存、重载逐值一致、增量写入后重新打开，以及拒绝坐标被修改后的写入。56 项集成测试和 6 项位置注意力测试通过。实际 DeepSeek 第二组候选尚未进入标准检查点。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$inputs = @('--checkpoint', 'runs/spora-native-query-key-v1/bootstrap.pt',
  '--reference', 'runs/spora-partial-query-key-v1/verified.json',
  '--attention', 'runs/spora-grouped-absolute-attention-v1/attention.pt',
  '--ceval', 'runs/ceval-validation-617524a')
.\.venv\Scripts\python.exe -u scripts/audit_spora_query_key_expansion.py @inputs --second-group-pairs 12 8 4 --output runs/spora-query-key-expansion-v2/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_query_key_expansion.py @inputs --second-group-pairs 8 --calibration-steps 96 --output runs/spora-query-key-local-alignment-v1/result.json
```
