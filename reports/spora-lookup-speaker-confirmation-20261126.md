# 转述中的说话人和确认意图

上一轮保存提问动作后，模型会把「同伴同意了，我还在考虑」误判为取消。本轮借鉴 SpeakerMem-R1 的说话人归属评测，在 13 种原训练答复上加 12 种对照。每组保留相同的他人立场，改变用户自己的肯定、否定或犹豫。两组固定同一个孢子候选、前 12 层读出以及训练设置，共用 14 种校准答复。语义读出位于两个循环注意力替换之前。状态更新沿用上轮保存的提问动作。 [SpeakerMem-R1](https://arxiv.org/abs/2609.26780) · [上一轮](spora-lookup-stored-confirmation-20261126.md)

| 指标 | 原训练集 | 增加说话人对照 |
| --- | ---: | ---: |
| 校准得出的阈值 | 0.90 | 0.95 |
| 转述新题，肯定与否定判对 | 4/16 | 10/16 |
| 转述新题，尚未决定时保留待确认 | 8/8 | 8/8 |
| 转述新题，明确指定动作判对 | 2/4 | 0/4 |
| 转述新题，从非空范围开始取消后清空 | 36/90 | 45/90 |
| 转述新题，取消后仍读出旧记录 | 6/120 | 6/120 |
| 转述新题，应读时读对记录、版次和值 | 12/60 | 30/60 |
| 旧题，应读时读对 | 60/96 | 36/96 |
| 上一批新措辞，应读时读对 | 96/96 | 36/96 |

转述新题使用 14 种新答复，分别搭配要求继续和要求取消的提问。每个问答组合覆盖三组地址和四种起始范围，共 336 次实际档案回放。其余两套各包括 480 次。地址、待确认动作和版次由评测提供。三组测试的尚未决定答复均保留了待确认，说明扩充校准集本身已能改善上一轮的提前清空。

在相同的 0.90 阈值下回看转述新题，独立答复分类从 7/14 提升到 9/14，其他两套分别从 7/10 降到 5/10、从 9/10 降到 6/10。这是使用保存分数的诊断。训练增加了针对相对答复的样本，没有增加用户直接指定继续或取消的样本。下一轮优先将答复类型和明确动作分开校准，用匹配的直接动作样本检验能否兼顾两类答复。当前候选留在离线实验。

运行约 106 秒，受检权重和档案摘要一致。逐题输出和两份读出权重位于 `runs/spora-lookup-speaker-confirmation-v1`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_confirmation runs/spora-lookup-transitions-clauses-v1 runs/spora-lookup-reset-matched-v1 runs/spora-lookup-speaker-confirmation-new --recurrent @weights --stored-action --speaker-contrasts
```
