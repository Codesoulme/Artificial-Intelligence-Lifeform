# 循环层替换对口语查阅确认的影响

前轮改变分句措辞和位置，未改善确认后的实际读回。已有受控字段实验发现，交换第 22 层的表示，可以改变之后部分注意力层查哪个字段。本轮固定训练数据和地址，直接检验读过两处循环层之后，确认头是否能借用更多请求信息。*Who Asks, Who Answers* 研究了循环表示与后续精确注意力的分工。*Multilinguality in Hybrid Attention LLMs* 则观察到多语言表示随混合层的顺序而改变。 [前轮确认结果](spora-confirmation-scope-transfer-20261126.md) · [孢子查询交换](spora-native-attention-memory-20261126.md) · [混合查询研究](https://arxiv.org/abs/2611.01909) · [多语言表示研究](https://arxiv.org/abs/2609.35378)

## 对照

加载同一份孢子父检查点，对同一批中文请求分别使用原注意力与第 19、22 层循环注意力。两组读到第 22 层输出为止，层号从零计。保留前轮的全句加末分句汇聚。两组均逐条处理无填充输入，用同一初始化、64 对训练句、96 条普通训练请求和已固定的校准规则训练确认头。回放阶段，两组在同一个循环版聊天实例中使用相同地址头和档案。回放按请求重置对话状态，并核对模拟用户确认后读取的实际编号和值。

## 结果

| 指标 | 原注意力表示 | 两处循环替换后的表示 |
| --- | ---: | ---: |
| 前批正例实际读对 | 2/8 | 2/8 |
| 新批正例实际读对 | 1/8 | 1/8 |
| 选错地址或对非查阅请求提议 | 0 | 0 |
| 确认前读取 | 0 | 0 |
| 跳过公开普通请求 | 128/128 | 128/128 |
| 校准阈值 | 0.95 | 0.95 |

两组放行的是相同的三条正例，前批与新批的最高逐题动作分数差分别为 0.087 和 0.034。首个循环替换之前，新批的特征逐元素相同。读过两处替换后，新批特征的平均绝对差为 0.0209。实验约 117 秒，源权重和档案摘要一致。62 项版次测试和 10 项循环注意力测试通过。逐题分数、回放及两份训练结果保存于 `runs/spora-confirmation-recurrent-features-v1/`。

第 19 层旧确认头在同批题上分别读对 3/8 和 3/8，也有 1 次和 3 次错提议。此轮相同深度的两种注意力都产生相同的查阅行为。样本只覆盖同一档案的 16 条应查题和 8 条其他请求，暂不能外推到开放中文对话或更深的骨干替换。

下一轮优先检查「明确要查，但人物或事项未定」能否成为单独的动作。先用独立样本区分确有疑义、可由前文消解和普通对话，比较问一个具体问题、给出完整地址等确认以及继续聊天三种行为。评分到用户补答后的实际读取，同时计多余追问和读错。ClariEdge 报告了可由上下文消解的难负例对澄清训练的作用，DiscoBench 分开评价疑义识别、提问质量和补答后的完成情况。 [ClariEdge](https://arxiv.org/abs/2610.28686) · [DiscoBench](https://arxiv.org/abs/2606.27669)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_confirmation_recurrent_features runs/spora-confirmation-scope-transfer-v1 runs/spora-crosswoz-field-transfer-v1 runs/spora-dense-joint-address-v1 runs/research/xing-usp-883c7dc runs/spora-confirmation-recurrent-features-new --recurrent @weights
```
