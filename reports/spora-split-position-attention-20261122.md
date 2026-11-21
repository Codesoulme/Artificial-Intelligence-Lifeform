# 第 18 层按键值头组迁移位置编码

以已经重载验证的第 26 层门控检查点作对照。第 18 层有两个键值头组，每组由一个键值头和六个查询头组成。本轮从已训练的绝对位置模块取出各组的 QKV，分别测只切换第 0 组、只切换第 1 组。另一组沿用原有 QKV 和旋转位置，两组共享一条注意力计算路径，每个词元仍只写一份键值。末端输出投影沿用父模型。

第一个对照复用训练好的权重。双位置路径分别计算投影，然后按头组选择进入注意力，原型增加一次投影。先测八个普通提示的末位置全词表 KL，再对每个单组候选测 60 道 C-Eval、八个三轮开发场景、32 条当前或历史记忆答案。两组全切换的词表 KL 用作整层对照。另用一条普通提示和一条显式记忆提示比较开启及关闭键值复用时的生成词元。

## 单组切换结果

| 候选 | 末位置平均词表 KL | C-Eval | 与父模型相同的对话词元序列 | 记忆短答 |
|---|---:|---:|---:|---:|
| 第 0 组 | 0.00220 | 23/60 | 12/24 | 32/32 |
| 第 1 组 | 0.00954 | 22/60 | 6/24 | 32/32 |
| 两组全切换 | 0.01323 | 21/60 | 6/24 | 32/32 |

父模型 C-Eval 为 `24/60`。两个单组候选均通过两条生成复用检查。全切换的 C-Eval、对话与记忆数字取自上轮实验。按八个普通提示的词表 KL 选择第 0 组进行整网校准。校准使用冻结父模型作为教师，只更新绝对位置路径的 QKV 和位置表，并在 16 个额外的普通与记忆请求上对照末两位置的词表分布。

## 整网校准

加入 32 条普通序列和 16 条记忆请求做 40 步校准后，16 条额外请求的平均 KL 从 `0.00598` 降为 `0.00561`。另八条普通提示的 KL 从 `0.00220` 上升到 `0.00356`。C-Eval 保持 `23/60`，与父模型逐词元一致的回答从 `12/24` 减少到 `9/24`，自然结束从 `19/24` 减少到 `17/24`。改约场景的最终回答准确保留了更新后的时间和地点，但前两轮加入无关建议并达到长度上限。32 条受控记忆回答与两条生成复用检查继续通过。

校准后的权重作为独立研究候选保存。下一轮先拆开查询与键的位置计算、值投影这两个改动来源，再按最终词表分布和中文任务的结果定位该保留的分量。当前两轮候选的选择题均低于父模型，结构迭代以父检查点为起点。

单组对照运行约 310 秒，校准及复测约 230 秒，各自峰值显存分配约 4.95 GiB。两个头组的路由和单路复用通过 CPU 测试，连同集成与结构迁移共 42 项测试通过。逐题报告保存在 `runs/spora-split-position-attention-v1/`。校准权重 `calibrated.pt` 的 SHA-256 为 `9e07d1eabdaf3b761041dbf6f9c981768735cb908c5d5ea1e7b6d69ee29f1b8e`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe scripts/audit_spora_split_position_attention.py --checkpoint runs/spora-language-migration-layer26-v1/candidate.pt --attention runs/spora-grouped-absolute-attention-v1/attention.pt --baseline runs/spora-language-migration-layer26-v1/migration.json --reload-report runs/spora-language-migration-layer26-v1/reload.json --ceval runs/ceval-validation-617524a --output runs/spora-split-position-attention-v1/result.json
.\.venv\Scripts\python.exe scripts/audit_spora_split_position_attention.py --checkpoint runs/spora-language-migration-layer26-v1/candidate.pt --attention runs/spora-grouped-absolute-attention-v1/attention.pt --baseline runs/spora-language-migration-layer26-v1/migration.json --reload-report runs/spora-language-migration-layer26-v1/reload.json --ceval runs/ceval-validation-617524a --calibrate-from runs/spora-split-position-attention-v1/result.json --output runs/spora-split-position-attention-v1/calibrated.json
```

## 相关迁移研究

CARE 的迁移目标是让低秩近似保住实际输入激活，并把更多表示能力分给更敏感的层。MLAfication 也按分量对注意力的贡献决定保留哪些旋转维度。两者的目标都是改变注意力表示。本项目可用固定的中文和记忆回放，直接衡量一个头组或若干旋转维度对最终词表分布的影响，再决定迁移顺序。 [CARE](https://arxiv.org/abs/2603.17946) · [MLAfication](https://arxiv.org/abs/2610.17202)
