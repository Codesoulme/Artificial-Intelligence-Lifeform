# 会话条件的字段与格式预测

在冻结的 DeepSeek 孢子检查点上，用前 12 层的问句特征分别预测字段及顺序、日期与钟点是否合写、输出样式。评测提供场景类别，程序按预测结果读取工作状态并排版。这个实验用于定位错误，排版阶段由程序完成。训练用 2,812 条合成问句。12 道新题不含与训练完全相同的问句。

| 评测条件 | 逐字通过 |
| --- | ---: |
| 自动写入与预测方案 | 9/12 |
| 正确状态与预测方案 | 11/12 |
| 正确状态，另提供正确字段及顺序 | 12/12 |
| 正确状态，另提供正确分组或正确样式 | 各 11/12 |
| 自动写入与完整的正确方案 | 9/12 |

在这批题上，字段及顺序预测 11/12，合写与样式均为 12/12。自动写入的三道失败分别是把「周日」截成「日」、漏掉「十六卷」以及保留旧装备。旧装备那道题还把场地误选成方向。此前的问句读出用的是另一批题和逐词生成，得分不可直接比较。

首次运行在反向传播时耗尽显存。重跑把问句特征留在系统内存，每批传入 24 条。训练与核验约 158 秒，峰值显存分配 4.94 GiB。核对了输入与权重摘要、个体摘要和十条日期记忆结果，四项工作状态单元测试通过。

下一轮需要把字段规划接回模型生成路径，在相同新题上比较程序排版和模型答复，并测量常规对话的回归。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$parent = 'runs/spora-dialogue-correction-v1/student'
.\.venv\Scripts\python.exe -u scripts/audit_spora_read_plan.py --checkpoint "$parent/bootstrap.pt" --database "$parent/individual.sqlite" --writer runs/spora-dialogue-writer-transfer-v1/writer.pt --output runs/spora-dialogue-read-plan-v1-minibatch
```
