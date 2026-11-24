# 第 19 层循环注意力迁移

在完成三处前馈门控迁移的 DeepSeek 孢子候选上，替换第 19 层的词元混合方式。近期两个 8 词元块保留精确因果注意力，更早的块折入固定大小的矩阵和归一化向量。模型根据当前查询学习两路向量的混合权重。冻结已有的查询、键、值与输出投影、语言骨干和个体状态。训练顺序采用 MOHAWK 的思路，先对齐局部混合结果，再对齐整网预测。 [MOHAWK](https://arxiv.org/abs/2408.10189) · [Llamba](https://arxiv.org/abs/2502.14458)

用 36 段中文材料拟合，留出 12 段中文材料和 12 段多轮更正历史。局部对齐训练 160 步，整网预测对齐再训练 32 步。整网指标取每段最后一个位置，比较与父模型的全词表分布及首选词元。

| 第 19 层条件 | 留出混合相对平方误差 | 全词表平均 KL | 首选词元一致 |
|---|---:|---:|---:|
| 只保留近期注意力 | 1.01741 | 0.25251 | 19/24 |
| 加入随机初始化的循环状态 | 0.80433 | 0.19220 | 19/24 |
| 对齐局部混合 | 0.15982 | 0.04203 | 23/24 |
| 再对齐整网预测 | 0.15847 | 0.04068 | 23/24 |

独立进程重载循环层，将其接入原模型的增量生成。对 94、190、382 词元的输入，以预填充加末尾四次单步解码，与一次性计算对照。该层每次都保存 14 个近期词元和一份固定大小的摘要，张量合计 320,640 字节。三个长度的最后四个首选词元一致，最大对数分数差分别为 0.0215、0.0278、0.0366。其余解码层继续使用各自原来的状态，因此数字只计算第 19 层。现场尚无生成延迟对照。

在原本 12 道更正题的相同可见历史上，父模型完整通过 3 道，候选通过 4 道。多通过的一题是正确连续写出「纸鸢园门」。这套题也参与过留出诊断，范围限于三个任务。32 条受控持久记忆答复逐项一致，个体摘要一致。真实生成测试的峰值 CUDA 分配为 3.40 GiB。候选保存在 `runs/spora-gelu24-recurrent-layer19-v2/`。实验缓存支持单序列贪心生成，无法撤销已经并入循环状态的历史。

```powershell
$sporaBase = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaRevision = 'runs/spora-gelu24-write-revision-v1/comparison.json'
$sporaLayer = 'runs/spora-gelu24-recurrent-layer19-v2'
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_attention $sporaBase $sporaRevision $sporaLayer
.\.venv\Scripts\python.exe -m scripts.verify_spora_recurrent_attention $sporaBase $sporaRevision $sporaLayer
```
