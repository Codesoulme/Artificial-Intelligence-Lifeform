# 孢子普通中文与日期记忆交互试验

日期入口增加 `--chat`。同一个 DeepSeek 孢子检查点处理普通回复和神经记忆读取。普通问题采用底座对话模板、直接贪心生成、单轮最多 64 词元。最多保留六组最近的普通问答，超出位置上限时从最早的完整问答开始舍弃。日期回忆沿用独立的人物与时间坐标。回忆结果不会写进普通问答的提示，普通问答也不向解码器供给持久记忆张量。形似记忆请求但无法定位的句式进入澄清。

四个开发场景共 21 轮，覆盖周五计划、凝露解释、改写与不确定信息。五条日期读取正确，五条澄清或拒绝符合预期。模型原样生成了 11 条普通回复，其中 10 条自然结束。包含记忆的问答开始和结束时个体摘要一致，原数据库及检查点的文件摘要一致。逐轮记录见 `runs/spora-mixed-calendar-chat-v1/result.json`，实测约 97 秒，峰值 CUDA 分配约 4.94 GiB。

普通回复能复述周五计划，跨两次独立记忆读取后仍答出图书馆，也按要求先把火车改成公交车。失败输出包括把免费备选答成低价咖啡馆、解释凝露时未交代低温杯壁、继续改写时恢复了已替换的火车，以及把询问昨晚到家时间改成了询问预计到家时间。普通轮次仍需按独立开放对话标准验收。

## 固定对话历史的原版对照

原版 DeepSeek 使用相同的用户问句、相同的历史普通回复、同一模板和相同生成预算。11 条中有 7 条可见回复逐字一致。免费备选答成低价选项、误解凝露和恢复旧改写在原版也出现。询问室友到家时间这题，原版保留了过去时，孢子改成了将来时。原版 9/11 自然结束，孢子 10/11。当前对照说明两份模型共享多项错误，结构迁移后仍可能产生逐题差异。原版和孢子逐题对照见 `runs/spora-mixed-calendar-chat-v2/result.json`，含整套复测约 115 秒。

隔离源码通过 72 项回归，并独立启动命令行实测五轮切换。周五计划、两条日期记忆、回到普通追问和有歧义的双日期请求均走预期入口，保存的个体文件摘要一致。控制台记录见 `runs/spora-mixed-calendar-chat-v2/cli.jsonl`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$source = @('--checkpoint', 'runs/spora-native-query-key-v1/bootstrap.pt', '--database', 'runs/spora-calendar-sleep-v1/individual.sqlite')
.\.venv\Scripts\python.exe -u scripts/spora_calendar_history_chat.py @source --chat
.\.venv\Scripts\python.exe -u scripts/audit_spora_mixed_calendar_chat.py @source --output runs/spora-mixed-calendar-chat-example/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_mixed_calendar_chat.py @source --compare-original runs/spora-mixed-calendar-chat-example/result.json --output runs/spora-mixed-calendar-chat-comparison/result.json
```
