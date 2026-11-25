# 字段级读取与排除的实际回放

上一轮的整句读取判断能压掉两次误读，却无法处理一句话里「用已有饮品，另选书」这类混合意图。本轮在同一份冻结 DeepSeek、同一批候选描述和同一份版本档案上，对照逐字段二分类和三分类。三分类区分要求读取、明确排除和其余情况。参照 Schema-Guided Dialogue 按字段提供描述，并用 Hy-MultiTurn 的精确对象和约束整合、MemConflict 的冲突题思路设计评测。 [上一轮](spora-read-authority-20261126.md) · [Schema-Guided Dialogue](https://arxiv.org/abs/1909.05855) · [Hy-MultiTurn](https://arxiv.org/abs/2607.29196) · [MemConflict](https://arxiv.org/abs/2605.20926)

两种新头用相同的 372 条请求、相同冻结特征和相同步数。原有 156 条标注只有应读字段，其余统一作为背景类。新增 216 条通过互换字段、语序和说法明确标注排除项。二分类把排除与背景合并。二分类的读入阈值保持为 0.5，三分类只有最高分为应读才采用。旧候选、二分类和三分类使用同一已有显式版次路径，分别执行选中的张量读出。测试运行在六位虚构人物的 24 份版次张量上，确认请求中未出现存储值，并按完整字段集合和最终生成的值分别计分。

| 指标 | 旧字段头 | 同数据二分类 | 同数据三分类 |
| --- | ---: | ---: | ---: |
| 旧 23 条完整读对 | 7/13 | 7/13 | 7/13 |
| 上轮新增 20 条完整读对 | 3/11 | 3/11 | 2/11 |
| 新增 18 条排除题完整读对 | 2/12 | 4/12 | 5/12 |
| 新增排除题中被读出的排除字段数 | 4 | 3 | 4 |
| 三组无需读取却读出的请求 | 5/25 | 4/25 | 6/25 |
| 三组合计实际生成完整且准确的事实值 | 12/36 | 14/36 | 14/36 |

「书名不必写，只找会面地点」一题，旧头读出了两个值，两种新头均只读回地点。遇到「地点那一栏留空，只用上回确认带的东西」，两种新头都只读回了应该排除的地点。三分类还在明确另选地点和物件的请求中读回了旧物件。标签更细没有稳定改善读取边界。同数据二分类在当前样本中误读较少，但完整读取仍不足一半。七条显式版次回归全部通过，版次档案摘要一致，49 项版本测试通过。事实读回检查要求内容值匹配和自然结束，尚未评定口语成品是否自然可用。

首轮用时约 93 秒，CUDA 张量峰值约 3.47 GiB。独立进程用时约 55 秒，复现 61 条自然请求中所有路径的记录选择和回答，以及七条显式回归。逐题证据位于 `runs/spora-field-authority-v1/`。

两种新头留在研究目录。下一轮优先给每个地址候选绑定实际支持短语，单测「只读这一项」「排除旧项」「提及已登记者作为反例」三类相近请求。使用新校准题和留出题，避免用这 18 条调阈值。已有版次入口可继续验证完整消息的重启恢复。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaRun = 'runs/spora-field-authority-new'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_field_authority runs/spora-composition-training-v1 runs/spora-dual-version-reply-v1 runs/spora-field-evidence-v1 $sporaRun
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_field_authority runs/spora-composition-training-v1 runs/spora-dual-version-reply-v1 runs/spora-field-evidence-v1 $sporaRun --verify
```
