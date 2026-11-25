# 同题对照中文指令底座与孢子改造版

前两轮在 DeepSeek 末层训练小型残差。冻结底座通过 3/8 道短改口题，监督残差通过 2/8，增加预测保持后通过 1/8。两种残差各丢失两道原本正确的题。暂停这条训练路线，改用本机已有指令底座做同题对照。 [残差对照](spora-reply-preservation-20261125.md)

复用原轮锁定的 12 道 FollowBench 中文格式题与 8 道自编改口题。比较原版 Qwen2.5-1.5B-Instruct、载入同一套权重的孢子改造版，以及上一轮留存的冻结 DeepSeek-R1-Distill-Qwen-1.5B 直接作答。问题、可见历史、贪心解码和评分器一致。模型使用各自的聊天模板，格式题与改口题分别限制在 128 和 72 个生成词元。两种千问配置均只测普通语言生成。 [固定版公开评测](https://github.com/YJiangcm/FollowBench/blob/6278f4c1377b4eafab737267b8b21acd52ea0e52/data_zh/format_constraints.json)

| 指标 | DeepSeek 原版 | 千问原版 | 千问接入孢子 |
| --- | ---: | ---: | ---: |
| 短改口题通过 | 3/8 | 6/8 | 6/8 |
| 短改口题正常结束 | 8/8 | 8/8 | 8/8 |
| 公开格式题通过规则筛查且结束 | 1/12 | 0/12 | 0/12 |
| 公开格式题触及词元上限 | 9/12 | 11/12 | 11/12 |

两种千问都保住了 DeepSeek 答对的三题，额外完成换地点、换用品与取消提醒。逐题复核中，两种千问都把「删掉拍照，保留时间、地点和手套」答成了处理确认，没有交付改写后的通知。更改借球拍时间时，原版擅自把周四改成周六，接入版漏掉星期且写错了时间。两种千问在改口题中有 7/8 条可见答复逐字相同，在格式题中有 7/12 条相同。

长格式题受生成预算限制。例如两版千问的胶片文章都在论点中间截断。也有截断前已经出现的问题，例如煎鲑鱼的请求被回答成烤鲑鱼，HTML 行程使用了数字标签。DeepSeek 唯一的格式题初筛通过项在人工复核中仍有句长缺陷。此组只能描述本机短预算表现。公开集也无法排除模型预训练时接触类似内容。

千问两路共用约 326 秒，CUDA 分配峰值 2.96 GiB，含加载和核对。脚本检查模型、检查点、校准文件和重建后的个体状态摘要。6 项普通作答评估测试和 49 项版次测试通过。逐题输出保存在 `runs/spora-reply-backbone-control-v1/result.json`。

下一轮先在符合任务长度的预算下比较新模型原版，优先验证独立三至五轮能否交付可用的成品。只有原版取得足以抵偿迁移成本的提升，才安排接入。Qwen3.5-0.8B 的固定架构含 18 层线性注意力和 6 层完整注意力，官方模型卡要求新版 Transformers。它是单独测量中文与延迟的候选，本轮只核对了官方资料。本机版次入口仍使用 DeepSeek。任何新入口还需在空对话及独立进程中验证模型内张量换值和读回。对话判分单列新要求、废弃内容和成品可用性。 [Qwen3.5-0.8B 模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) · [AFVE-Bench](https://arxiv.org/abs/2611.10541)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaData = 'runs/research/followbench-6278f4c/format_constraints_zh.json'
$sporaReference = 'runs/spora-reply-preservation-v1/result.json'
$sporaCheckpoint = 'runs/named-memory-layer9-packed-reply-v1/packed.pt'
$sporaDonor = 'D:/SPORA-runtime/models/Qwen2.5-1.5B-Instruct-989aa798'
$sporaCalibration = 'runs/named-memory-qwen-structure-control-v2/result.json'
$sporaOutput = 'runs/spora-reply-backbone-control-fresh/result.json'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_reply_backbone_control $sporaData $sporaReference $sporaCheckpoint $sporaDonor $sporaCalibration $sporaOutput
```
