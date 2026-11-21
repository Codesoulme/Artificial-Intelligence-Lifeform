# 第 18 层的部分查询和键位置迁移

从已完成增量恢复验证的值位置检查点出发，在第 18 层第 0 组试验部分查询和键维度。每个头有 128 个坐标，Qwen2 的旋转位置将它们分成 64 对。每次替换成对的两个坐标，并让同组的六个查询头与一个键头在这些坐标上使用同一张孢子可学习绝对位置表。权重取自此前训练过的绝对位置投影。其余坐标继续按原方案旋转，值路径使用已纳入标准检查点的实现。

筛选比较最低频的 8、16、32 对。在八个普通首轮提示上，另用最高频 16 对作词表分布对照。每个低频候选都在 60 道 C-Eval 上与父模型逐题比较。追加评测优先验答对数较高的候选，同分时优先验替换更多坐标的候选。复测包含 83 道额外选择题、24 轮对话和 32 条受控记忆答案。另检查一条普通生成和一条记忆生成在开启和关闭键值复用时是否逐词元相同。组件按原模型的层、头组和父文件摘要绑定。

这轮按频率排序作为简单、可复现的初始尝试。MLAfication 的部分旋转位置方法还衡量各分量对注意力的贡献，为后续选择敏感维度提供了另一种依据。 [MLAfication](https://arxiv.org/abs/2610.17202)

## 筛选

| 查询与键的替换范围 | 八条提示的末位置词表 KL | C-Eval | 与父模型选项相同 |
|---|---:|---:|---:|
| 最低频 8 对 | 0.000103 | 25/60 | 59/60 |
| 最低频 16 对 | 0.000739 | 25/60 | 59/60 |
| 最低频 32 对 | 0.001045 | 24/60 | 58/60 |
| 最高频 16 对 | 0.001116 | — | — |

父模型答对 24/60。初筛用逐题一致作硬门槛，把改对一道题的两个候选也拦下了。复测改用答对数作初筛，同时记录逐题差异，让这两个候选都具备完整评测资格。复测检查两组选择题答对数、对话自然结束、受控记忆和生成复用。

## 16 对的完整评测

最低频 16 对通过完整评测，覆盖第 0 组每个查询和键头的 32 个坐标。三科答对 `25/60`，其中一题逻辑由错改对。额外四科答对 `23/83`，一题物理由原来的错选项换成另一错选项。24 轮对话中 20 轮逐词元一致，两边都是 18 轮自然结束。变动集中在早餐和雨天计划，对话仍存在限制遵循错误。32 条受控记忆答案逐项一致，两条生成复用检查通过。8 对和 32 对本轮只做了筛选。

选择的参数保存在 690,200 字节的 `selection.pt`，加载时与已有的标准检查点配合使用。该文件只保存部分查询与键投影，向量表由父检查点提供。

新进程读取该组件后，复现了临时迁移时的全部 143 道选择题选项、24 轮生成、32 条记忆答案和两条生成复用检查。组件 SHA-256 为 `36d61a7e2455de6392d880103be7134a25183ea9dd987b56e380607f801c4088`。三段本机运行分别约 121、200、173 秒，峰值显存分配约 4.94 GiB。相关 CPU 测试 39 项通过。保留该组件作为后续查询与键改造的起点，普通聊天仍需要独立新场景评测。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$checkpoint = 'runs/spora-native-value-checkpoint-v1/bootstrap.pt'
$reference = 'runs/spora-packed-value-position-v1/verified.json'
$ceval = 'runs/ceval-validation-617524a'
$destination = 'runs/spora-partial-query-key-v1'
.\.venv\Scripts\python.exe scripts/audit_spora_partial_query_key.py --checkpoint $checkpoint --reference $reference --ceval $ceval --attention runs/spora-grouped-absolute-attention-v1/attention.pt --output "$destination/selection.json"
.\.venv\Scripts\python.exe scripts/audit_spora_partial_query_key.py --checkpoint $checkpoint --reference $reference --ceval $ceval --component "$destination/selection.pt" --output "$destination/verified.json"
```
