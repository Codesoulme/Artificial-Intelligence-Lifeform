# 部分查询与键的标准检查点和自动记忆评测

将第 18 层第 0 组的 16 对查询与键投影写入标准孢子检查点，与已经迁移的值路径共享同一张可学习绝对位置表。保存时只新增查询与键的投影及坐标清单。加载时检查层、坐标范围、投影宽度和对应值组件，再恢复这组参数。父检查点现有张量及元数据逐项校对。

自动记忆评测使用同一份姓名头、时间头和检查点内的回答层。父模型与重载候选都对同一批口语问句运行两份不同内部记忆状态，单条通过要求模型自行选对地址和时间版本，并随状态正确换值。评测覆盖 72 条有登记资料的问句和 40 条应拒绝读取的问句。姓名与时间头读取下标为 6、18 的隐藏状态，本轮第 18 层的修改位于这两处特征之后。对照还逐题检查实际生成是否随结构变动。

## 新进程结果

| 配置 | 地址正确 | 时间版本正确 | 自动换值通过 | 负例触发读取 |
|---|---:|---:|---:|---:|
| 值位置父模型 | 56/72 | 44/72 | 34/72 | 0/40 |
| 查询、键和值位置候选 | 56/72 | 44/72 | 34/72 | 0/40 |

112 条的路由和回答记录逐项一致。34 条同时选对地址与时间版本的问句全部通过换值。24 条历史问句中有 12 条被误选为更正后的版本，另有 16 条正例在姓名选址环节失败。当前数据是已有的受控饮品任务，可用来确认新结构下保留了自动查询行为。后续改善优先看历史版本消歧和姓名边界。

标准检查点为 26,198,860 字节，SHA-256 为 `f506d8c9f8a0ac0ca81b74fe8388c9af9fdb6bb8d2cb028ed929b0882cc468f0`。保存的查询与键权重和偏置逐项等于经过验证的源组件，标准加载时直接从此检查点恢复。父模型和候选评测分别约 160 秒、142 秒。干净源码下 39 项相关测试通过。逐题记录见 `runs/spora-native-query-key-v1/`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$arguments = @(
  '--parent', 'runs/spora-native-value-checkpoint-v1/bootstrap.pt',
  '--route-source', 'runs/qwen-compatible-gelu-layer17-v1/integrated.pt',
  '--router', 'runs/qwen-compatible-gelu-lexical-boundary-v1/hybrid',
  '--temporal', 'runs/qwen-compatible-gelu-temporal-factorized-v1/factorized',
  '--output', 'runs/spora-native-query-key-v1'
)
.\.venv\Scripts\python.exe scripts/audit_spora_native_query_key.py @arguments --component runs/spora-partial-query-key-v1/selection.pt
.\.venv\Scripts\python.exe scripts/audit_spora_native_query_key.py @arguments
```
