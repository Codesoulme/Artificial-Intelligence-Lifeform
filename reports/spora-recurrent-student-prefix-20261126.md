# 三层循环布局的学生前缀蒸馏对照

对同一个第 19、22、24 层定向蒸馏候选续训。两个组都使用同一批中文更正请求、相同的成对扰动、每组 16 步和相同的反向 KL 目标。一个组每步用当前学生权重贪心生成答复前缀，再由冻结的完整注意力父模型对相同前缀评分。另一个组使用等长标注答复前缀。两组总输入均为 32,044 词元，完成部分 415 词元，32 个答复轨迹全部存在内容差异。只更新三处循环核，反向传播放大 256 倍后在 FP32 参数上还原梯度。实验对照了 OPAL 的学生前缀机制，使用短答复和本机预算。 [定向训练的初始权重](spora-hybrid-targeted-recovery-20261126.md) · [OPAL](https://arxiv.org/abs/2609.31947)

| 配置 | 短更正 | 约 700 词元 | 约 1,400 词元 | 开放题逐轮 | 具名读回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 两层循环基线 | 8/10 | 7/10 | 2/10 | 8/18 | 32/32 |
| 三层定向训练起点 | 8/10 | 6/10 | 1/10 | 3/18 | 32/32 |
| 学生生成前缀 | 8/10 | 6/10 | 1/10 | 4/18 | 32/32 |
| 等长标准答复前缀 | 8/10 | 7/10 | 1/10 | 4/18 | 32/32 |

两种方法分别补回一个不同的开放问答轮次。在这轮小预算上，学生前缀没有优于标准答复前缀。两组读回和结束信号逐条相同，源权重、个体摘要和逐步词元预算均已核对。训练和重新加载评测约 606 秒，每组 CUDA 张量训练峰值约 4.09 GiB。十项循环注意力单元测试通过。 [双层基线](spora-hybrid-language-recovery-20261126.md)

相同 10 条最长更正题上，完整注意力父模型答对 4 条，三层定向版答对 1 条，成功题没有重合。父模型输出正确新值 4 次、保留字段 6 次，三层定向版分别为 2 和 4 次。这组结果显示父模型有自身的错误，同时保留了部分迁移后遗失的检索能力。父模型逐题记录位于 `runs/spora-recurrent-dialogue-extended-v1.json`。 [父模型对照报告](spora-hybrid-layer-placement-20261125.md)

后续结构对照改为测试有限的精确保留位置。HOLA 使用循环写入时的预测残差挑选要保留的历史条目，适合检验迁移后遗漏事实的现象。第 24 层当前权重使用求和更新，尚无预测残差。先对齐新增的增量更新，使其与同层教师混合输出可比较，再在相同条目预算下比较按写入残差保留、按近期保留及只用循环状态。首先检查能否保留被更正值和第二字段，再看独立生成。这个设计将存储限制放在历史条目数上。 [HOLA](https://arxiv.org/abs/2607.02303) · [Taylor-Calibrate](https://arxiv.org/abs/2606.16429)

## 第 24 层增量更新的初始化对照

复用求和候选的 36 篇训练材料、24 条短留出、8 词元块、160 步混合输出对齐及 32 步输出概率对齐，只替换为已有的带门控块增量更新。两组短留出的最高概率词元都与教师匹配 24/24。增量更新的最终 KL 为 0.04019，求和对照为 0.03735。相对混合误差分别为 0.29845 和 0.26119。新候选在一条 93 词元的回放上，整段与分块输出最大差异为 5.96×10⁻⁷，近期窗口上限为 16 词元。来源、个体摘要与对照一致，CUDA 张量峰值约 3.39 GiB。这个候选提供了可比较的增量写入路径，精确条目保留与长答复生成仍需验证。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$run = 'runs/spora-hybrid-targeted-recovery-v1'
$layers = @("$run/targeted/layer-19", "$run/targeted/layer-22", "$run/targeted/layer-24")
.\.venv\Scripts\python.exe -u -m scripts.train_spora_recurrent_student_prefix runs/spora-composition-late-gates-20-23-24-v1/candidate.pt "$run/result.json" runs/spora-recurrent-student-prefix-v1 @layers
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_recurrent_attention runs/spora-composition-late-gates-20-23-24-v1/candidate.pt runs/spora-gelu24-write-revision-v1/comparison.json runs/spora-gelu24-recurrent-layer24-gated-morph-v1 --layer 24 --update-rule gated_block_delta
```
