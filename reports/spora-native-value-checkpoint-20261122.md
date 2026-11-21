# 值位置模块的标准检查点与增量恢复

把已验证的第 18 层第 0 组值位置参数纳入标准孢子检查点。检查点记录层、头组、位置表和该组值投影，沿用固定的 DeepSeek 底座。标准加载入口同时恢复兼容层、GELU 门控、记忆读取和回答层，并在装载时校验头组、张量形状和可用位置范围。打包前后，原检查点的全部权重和个体数据逐项相同。

增量保存通过现有 SQLite 事务记录个体的变化。结构参数随新的 `bootstrap.pt` 固定，数据库绑定该文件的摘要。设定初始事实后，分别在事务提交前和提交后终止写入子进程，用新进程检查修订和当前／历史回答。

## 标准重载

新进程仅用标准检查点和固定底座加载，重现了独立组件评测的 24 轮对话词元和 32 条受控记忆答案。值路径参数与打包时逐项一致。隔离数据库的第一次写入生成修订 1，更新 8 个张量，共写入 205,200 字节张量数据和 15,652 字节清单数据。

## 中断恢复

| 子进程退出时机 | 数据库重开后读取的修订 |
|---|---:|
| 写入清单后、提交前 | 1 |
| 提交后 | 2 |

新进程在修订 2 恢复第 18 层值位置参数，当前记忆答出「可乐」，更正前的记忆答出「牛奶」。两条问题的地址由评测提供。标准检查点大小为 25,509,584 字节，前后文件摘要均为 `dc4da4b35e3e10bb481c6b167d123290aa311529b8402f0141ac687c60bd32a5`。干净源码环境的 38 项检查点、增量保存和注意力测试全部通过。本机恢复记录在 `runs/spora-native-value-checkpoint-v1/recovered.json`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$component = 'runs/spora-packed-value-position-v1'
$destination = 'runs/spora-native-value-checkpoint-v1'
.\.venv\Scripts\python.exe scripts/audit_spora_native_value_checkpoint.py pack --parent runs/spora-language-migration-layer26-v1/candidate.pt --component $component --output $destination
.\.venv\Scripts\python.exe scripts/audit_spora_native_value_checkpoint.py create --component $component --output $destination
.\.venv\Scripts\python.exe scripts/audit_spora_native_value_checkpoint.py recover --output $destination
```
