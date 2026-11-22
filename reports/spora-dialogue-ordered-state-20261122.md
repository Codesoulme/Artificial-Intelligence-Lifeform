# 会话条件的顺序读出

上一轮给孢子标注正确状态后，换值训练仍只能让 2/12 道题的更新字段随状态变化，完整通过 0/12。此轮冻结 DeepSeek 底座和原写入头，单独训练一个循环读出模块。它根据已有字段和上一动作，预测从状态中复制哪个词元，或生成哪个连接词及结束符。

训练数据覆盖约见、包裹和集合路线三类任务。每类使用 192 个场景，分别以初始和更新状态监督答案，共 1,152 条序列。评测每类使用四个未训练的值组合和提问方式。只给解码器最后一次追问，之前的用户消息用于自动写入组构建工作状态。要求答案与参考文本精确匹配且模型主动输出结束符。

| 生成方式 | 正确状态 | 旧状态对应的答案 | 自动写入 |
| --- | ---: | ---: | ---: |
| 仅 DeepSeek，无工作状态 | 0/12 | — | — |
| 候选词元加分 16 | 3/12 | 3/12 | 1/12 |
| 候选词元加分 36 | 3/12 | 3/12 | 1/12 |
| 单独运行读出模块 | 12/12 | 12/12 | 6/12 |
| 读出与底座按 95% 和 5% 混合概率 | 12/12 | 12/12 | 6/12 |

概率混合组每道题在使用初始状态和更新状态时，回答都随状态正确切换。两种权重的加分法都还会重复时间，或在包裹题中提前结束。较高比例的混合结果与单独读出一致，因此本轮验证的是受控读出能力。连接词表来自三类训练答复格式，读出模块也没有独立判断任意问题是否适合使用该格式。下一阶段需要检验新答复形式，并处理自动写入遗漏或误写导致的六道失败。

两次独立进程从保存的读出权重重开，比较了不同组合策略。三次运行的既有神经个体摘要、五条日期读取和五条澄清与上一轮逐项一致。运行时长约 127 至 132 秒，峰值 CUDA 分配约 4.94 GiB。四项工作状态单元测试通过。

## 复现

```powershell
$parent = 'runs/spora-dialogue-correction-v1/student'
$writer = 'runs/spora-dialogue-working-state-v2/working-state.pt'
$common = @('--checkpoint', "$parent/bootstrap.pt", '--database', "$parent/individual.sqlite", '--writer', $writer)
.\.venv\Scripts\python.exe -u scripts/audit_spora_ordered_working_state.py @common --output runs/spora-dialogue-ordered-state-v1
$reader = 'runs/spora-dialogue-ordered-state-v1/reader.pt'
.\.venv\Scripts\python.exe -u scripts/audit_spora_ordered_working_state.py @common --read-from $reader --weight 36 --output runs/spora-dialogue-ordered-state-v2
.\.venv\Scripts\python.exe -u scripts/audit_spora_ordered_working_state.py @common --read-from $reader --mix .95 --output runs/spora-dialogue-ordered-state-v3
```
