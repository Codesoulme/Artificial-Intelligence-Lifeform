# 追问驱动的条件读出

在孢子的会话工作状态上连接一个问句条件化读出头。冻结 DeepSeek 底座及上一轮的写入权重。读出头使用底座前 12 层的问句特征和可用字段，逐步预测要读取的子词、分隔符、标签及结束符。解码继续使用 95% 读出概率与 5% 底座概率的混合。

第一组汇总问句特征均值，用 3,096 条序列训练 25 轮。第二组增加循环编码以保留问句词序，同时扩大格式改写并提高常规问法的采样比例，用 6,792 条序列训练 20 轮。两组同时改变了架构和数据配置，因此无法隔离词序编码带来的增益。

| 读出 | 原格式，正确状态 | 原格式，自动写入 | 新格式，正确状态 | 新格式，自动写入 |
| --- | ---: | ---: | ---: | ---: |
| 上轮固定顺序 | 12/12 | 11/12 | 0/12 | 0/12 |
| 问句均值 | 3/12 | 2/12 | 3/12 | 2/12 |
| 词序编码与常规问法回放 | 8/12 | 7/12 | 3/12 | 2/12 |

严格通过要求答案与参考文本逐字一致且正常结束。训练中没有与这两组评测完全一致的问句。第二组是在看过首轮表现后设计的，复用题目用于诊断，仍需在新的一批问法上确认效果。

第二组正确完成了按装备、方向、集合点顺序写一行的题目。仍会把日期和钟点分开输出，而题目要求把两者合成一项。也出现答对了数值但漏标签、或者问数量却回答姓名的情况。以现有准确率，候选保留为实验，原读出仍作回归对照。后续要分别监督字段选择、字段分组和表面格式，并对普通问法单独验收。

两组本机运行分别约 139 秒和 167 秒，峰值显存分配约 4.94 GiB。核对了权重与实验输入的摘要、个体摘要及十条日期记忆结果。四项工作状态测试通过。

## 复现

```powershell
$parent = 'runs/spora-dialogue-correction-v1/student'
$writer = 'runs/spora-dialogue-writer-transfer-v1/writer.pt'
$common = @('--checkpoint', "$parent/bootstrap.pt", '--database', "$parent/individual.sqlite", '--writer', $writer)
.\.venv\Scripts\python.exe -u scripts/audit_spora_question_read.py @common --output runs/spora-dialogue-question-read-v1
.\.venv\Scripts\python.exe -u scripts/audit_spora_question_read.py @common --ordered --epochs 20 --output runs/spora-dialogue-question-read-v2
```
