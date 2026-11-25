# 独立中文格式题与普通回复能力保持

上轮末层残差改善了一道简短答复，但把原本正确的周日改期改错。本轮用独立题目区分答对、补回和退步，并在相同普通语言残差上比较预测保持。学习保持采用冻结父模型的输出分布作参考。低秩适配的遗忘研究也建议把任务新增能力与原能力分别测量。 [上轮结果](spora-local-ordinary-reply-20261125.md) · [Learning without Forgetting](https://arxiv.org/abs/1606.09282) · [LoRA 能力变化](https://arxiv.org/abs/2405.09673)

## 独立评测

从固定版 FollowBench 中文格式文件选取六组、每组两个难度，共 12 题。包含胶片摄影文章、深度学习工具分层说明、技术问题处理、烹饪步骤、木桌养护及冲绳行程的 HTML。另编八个两轮改口对照，分别更改日期、数量、地点、物件或时间，或仅取消一项。对照中的原已确认内容固定，对三种模型使用相同上下文。公开题最多生成 128 词元，改口题使用本机入口的 72 词元。所有题在训练开始前锁定，公开题仅用于评估。 [固定版中文格式数据](https://github.com/YJiangcm/FollowBench/blob/6278f4c1377b4eafab737267b8b21acd52ea0e52/data_zh/format_constraints.json)

公开题统计指定结构、主题相关词、部分可形式化的字数限制和自然结束。原作者对多数格式题另有模型评分，本轮的规则计数只适用于所列题目。改口题检查改后值、仍有效的值、禁用旧值、输出格式和结束。另逐条阅读通过样本与关键差异。报告同时列出相对父模型的新增通过和原题失分。

## 训练对照

三组为冻结 DeepSeek、上轮已校验的监督残差、以及增加预测保持的残差。两个残差沿用秩 16、49,152 个参数、32 条监督样本、同一初始种子及 192 步采样顺序。新组另使用 12 条人工写定的修订示范，只保留父模型在标准答案前缀下首选词元恰好正确的位置，且排除纯标点。每步从中采三处，约束新残差的完整词表概率接近父模型。分布保持及正确答复的监督使用同一个冻结词表投影。新增示范与两套评测分开。

## 结果

| 留出指标 | 冻结底座 | 监督残差 | 预测保持残差 |
| --- | ---: | ---: | ---: |
| 公开题同时通过筛查与结束 | 1/12 | 0/12 | 0/12 |
| 公开题含主题所需词 | 11/12 | 4/12 | 4/12 |
| 公开题通过结构筛查 | 4/12 | 0/12 | 0/12 |
| 公开题正常结束 | 3/12 | 7/12 | 3/12 |
| 改口题字段及形式通过 | 3/8 | 2/8 | 1/8 |
| 改口保住父模型已通过的题 | 3/3 | 1/3 | 1/3 |
| 改口补回父模型未通过的题 | — | 1/5 | 0/5 |

两个残差都丢了两道日期修订。保持组能选对星期二，却漏掉需要携带的合同。另一个例子中只复述「把发送时间改到周一早上」，没有交付给冯老师的通知。两组保住了修改试卷数量的题。监督组补回只留拔电源的提醒，其他场景仍有重复或套用训练语气的失败。八道改口题三组全部正常结束，失分来自内容或形式。

公开题唯一通过初筛的是冻结底座的三段木桌养护。作者对该组的自动方法按原始句子长度计数。本轮复核去除标题和标点后，有四句少于或等于 20 个文字，因此该答复只算规则筛查通过。两组残差均没有完整实现公开题要求，新增预测保持的一组出现短语循环。全轮约 358 秒，CUDA 张量峰值约 3.44 GiB。

## 重载核验

独立进程重算得到同样的 101 个训练锚点，并从两个保存件恢复权重。监督残差的首选词元与父模型一致 95/101，平均分布 KL 为 0.247。增加保持损失后为 100/101，KL 为 0.008。这说明训练位置上的分布保持已生效。新提示下完成整条改期的能力仍然退步。重载在一个公开题和两个改口题逐词复现全部 9 个模型输出，约用 109 秒。6 项语言评估测试与 49 项版次记忆测试通过。

两个候选保存在研究目录。后续用这批不参与训练的题比较本机已有中文指令模型，检查相同生成预算下可达到的内容和格式水平，再决定普通语言部分采用哪些已核验的训练信号。版次记忆入口沿用冻结底座。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaData = 'runs/research/followbench-6278f4c/format_constraints_zh.json'
$sporaReference = 'runs/spora-ordinary-reply-residual-v1'
$sporaOutput = 'runs/spora-reply-preservation-v1'
New-Item -ItemType Directory -Force -Path (Split-Path $sporaData) | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/YJiangcm/FollowBench/6278f4c1377b4eafab737267b8b21acd52ea0e52/data_zh/format_constraints.json' -OutFile $sporaData
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_reply_preservation $sporaData $sporaReference $sporaOutput
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_reply_preservation $sporaData $sporaReference $sporaOutput --verify
```
