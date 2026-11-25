# DeepSeek 孢子的双层注意力迁移对照

现有孢子把第 19、22 层改成了近期精确注意力加循环历史，并保留具名个体记忆。多次调循环写入和训练目标，在约 1,400 词元处仍难以同时回答两个有效字段。接下来比较替换层的位置，在整网运行中检验这些层相互影响的结果。 [现有双层实现](spora-recurrent-attention-layer19-20261124.md) · [长前文多字段对照](spora-recurrent-field-composition-20261124.md)

11 月 18 日的 FlashMorph2 研究发现，用原骨干的隐藏状态单独评价各层，以及用训练前的权重给替换位置排序，都会漏掉真实组合的误差与补偿。它用固定预算采样离散层配置，执行整网前向，同时按后续恢复目标调整权重。论文实验使用 Qwen3 和线性注意力变体。孢子将验证其中两项可迁移的做法，即直接运行真实层组合，以及按恢复后的表现评价组合。 [FlashMorph2](https://arxiv.org/abs/2611.24809) · [论文引用的线性注意力实现](https://github.com/fla-org/flash-linear-attention)

另有近期论文在若干混合推理模型中观察到，低精度的循环状态会增加推理长度和耗尽预算的次数。孢子的现有循环历史用 FP32 累加，本地组间保持这一精度，并对照近期键值所用的 FP16。当前开放中文推理试验用的是另一份封装检查点，因此不拿它的长答现象归因于循环状态精度。 [循环状态精度研究](https://arxiv.org/abs/2611.12127) · [孢子循环注意力源码](../spora/research_recurrent_attention.py)

## 固定替换预算

先用第 19、22 层的原始对齐产物，与第 21、24 层的同规格初始对齐产物比较。层号从零开始，两组都替换两层，间隔同为三层。共同父检查点是 `runs/spora-composition-late-gates-20-23-24-v1/candidate.pt`，保留第 5、17 层的持久读取路径。两组使用 8 词元块、相同中文材料、160 步局部混合对齐和 32 步整网预测对齐。核对权重元数据后，父模型前馈门控所在层为 21、24、26。两组使用同一副冻结的门控权重。

局部对齐只用来初始化。随后对两种真实组合使用相同的成对样本、更新数和生成条件，在各自的整网传递下恢复。比较恢复前后的长前文合并作答、有效与过期值、32 条受控持久读取、实际状态大小和延迟。现有组合充当对照，额外报告完整注意力父模型的可达到水平。该试验验证两种层组合，论文中的全网联合搜索属于后续研究范围。

## 初始化结果

| 单层位置 | 局部对齐后的平均 KL | 再对齐整网预测后的平均 KL | 最终首选词元一致 |
| --- | ---: | ---: | ---: |
| 19 | 0.04203 | 0.04068 | 23/24 |
| 22 | 0.02769 | 0.02792 | 24/24 |
| 21 | 0.26597 | 0.31215 | 21/24 |
| 24 | 0.03830 | 0.03735 | 24/24 |

核对四个候选使用了同一父检查点、训练与留出来源和个体摘要。两处新位置各保存一份约 296 KiB 的权重，摘要与各自报告一致。93 词元分段对照中新层保持相同的 16 词元近期上限，块状与增量实现的最大绝对误差约为 `7.15e-7`。第 21 层后续预测对齐使单层 KL 上升。该结果说明要把它作为较难的初始化情况记录，真实双层组合的长前文表现由下一环节决定。

## 整网恢复前

| 配置 | 短题完整 | 约 700 词元完整 | 约 1,400 词元完整 | 开放题逐轮词面通过 |
| --- | ---: | ---: | ---: | ---: |
| 完整注意力父模型 | 8/10 | 4/10 | 4/10 | 10/18 |
| 循环层 19、22 | 8/10 | 7/10 | 2/10 | 8/18 |
| 循环层 21、24 | 6/10 | 1/10 | 1/10 | 6/18 |

两份基准中的父模型逐题输出与判分一致，题集、父检查点和重建个体摘要一致。约 700 词元时，第 21、24 层组合写出了正确的新值 3 次，应保留的字段只出现 2 次。第 19、22 层组合这两项均为 7 次。约 1,400 词元时，两组的保留字段分别为 2 和 5 次。开放题这一列沿用现有词面判分。

由于循环实现从旧训练以来增加了更新规则和执行路径，本轮在当前代码下重跑两个恢复条件。各训练 32 步，学习率为 0.0005，只更新两处循环分支。使用完整答案交叉熵和首个不同答案词元的对比损失。对照程序核对两组抽样索引与序列长度的逐步记录、恢复权重摘要和相同测试来源。 [成对更正训练设计](spora-recurrent-counterfactual-20261124.md)

## 匹配恢复后的生成

| 配置 | 约 700 词元原更正 | 约 1,400 词元原更正 | 约 700 词元换值题，单题 / 两边均对 | 约 1,400 词元换值题，单题 / 两边均对 | 具名持久读取 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 循环层 19、22 | 7/10 | 2/10 | 5/10 · 2/5 | 3/10 · 1/5 | 32/32 |
| 循环层 21、24 | 1/10 | 1/10 | 2/10 · 0/5 | 2/10 · 1/5 | 32/32 |

两组使用同样的 48 对训练样本，实际抽到 31 个不同的对。核对两组的 32 个抽样位置和输入长度逐步一致。两组的原更正和成对更正得分与恢复前一致。约 700 词元时，第 21、24 层组合在一条交接题上输出了两遍「十七捆」，漏掉收件人。第 19、22 层组合输出了收件人和更新后的数量。持久读取给定姓名对应的地址和版本，两组的 32 条完整生成结果逐条相同。训练及评测用时分别约 606、616 秒，训练的 CUDA 张量峰值分别约为 4.07、4.11 GiB。

## 本机成本

| 输入词元 | 两组循环版 / 父模型张量存储 | 19、22 层首词 / 同次父模型 | 21、24 层首词 / 同次父模型 | 19、22 层增量每步 / 同次父模型 | 21、24 层增量每步 / 同次父模型 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 640 | 16.56 / 17.50 MiB | 1,050 / 1,066 ms | 1,054 / 1,034 ms | 74.6 / 69.3 ms | 58.3 / 56.0 ms |
| 1,536 | 39.31 / 42.00 MiB | 4,377 / 4,489 ms | 3,340 / 3,395 ms | 66.7 / 68.8 ms | 61.6 / 55.5 ms |

两种组合在最长输入时各节省约 2.69 MiB，每个循环层均保存 164,992 字节的状态。两次独立计时中的父模型耗时也有明显波动，所以表中按各自同一轮父模型比较。增量耗时的变化方向随配置和输入长度而异。29 项循环注意力与持久化测试通过，另有两项被套件跳过。来源对照额外验证了摘要不一致会被拒绝，共用一处循环层的布局可以比较。

## 后续选择

目前保留第 19、22 层组合做结构基线。下一步先试第 19、24 层，保留较接近父模型的第 24 层，恢复第 21 层的完整注意力。若需进一步区分，再试第 21、22 层。每个组合仍替换两层，用同题比较未更改字段是否保留。两种交叉组合可以检查第 21 层的影响是否随搭档而变。层间距离和父模型前馈门控所在位置也在变化，结果解释以这四种组合为限。HALO 和 KL 引导的混合注意力研究都给出了选择替换位置的实测对照。 [HALO](https://arxiv.org/html/2601.22156v1#S6.T4) · [KL 引导的层选择](https://arxiv.org/html/2512.20569v1#A7.T14)

## 初始化命令

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaParent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaRevision = 'runs/spora-gelu24-write-revision-v1/comparison.json'
foreach ($sporaLayer in 21,24) {
    $sporaOutput = "runs/spora-gelu24-recurrent-layer${sporaLayer}-morph-v1"
    .\.venv\Scripts\python.exe -u -m scripts.audit_spora_recurrent_attention $sporaParent $sporaRevision $sporaOutput --layer $sporaLayer
    if ($LASTEXITCODE -ne 0) { break }
}
```

新组合的恢复前对照使用与第 19、22 层既有报告相同的题集与 2,048 位置预算。报告应先核对两次评测的完整注意力对照及来源摘要，再进入匹配更新数的恢复试验。

```powershell
.\.venv\Scripts\python.exe -u -m scripts.benchmark_spora_recurrent_dialogue $sporaParent runs/spora-recurrent-morph-21-24-extended-v1.json runs/spora-gelu24-recurrent-layer21-morph-v1 runs/spora-gelu24-recurrent-layer24-morph-v1 --dialogue-only --extend
```

## 恢复与对照命令

```powershell
$sporaOldRef = 'runs/spora-recurrent-dialogue-extended-v1.json'
$sporaNewRef = 'runs/spora-recurrent-morph-21-24-extended-v1.json'
$sporaOldRun = 'runs/spora-morph-recovery-19-22-v1'
$sporaNewRun = 'runs/spora-morph-recovery-21-24-v1'
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_placement $sporaOldRef $sporaNewRef
.\.venv\Scripts\python.exe -u -m scripts.train_spora_recurrent_counterfactual $sporaParent $sporaNewRef $sporaNewRun runs/spora-gelu24-recurrent-layer21-morph-v1 runs/spora-gelu24-recurrent-layer24-morph-v1 --steps 32 --learning-rate .0005
.\.venv\Scripts\python.exe -u -m scripts.train_spora_recurrent_counterfactual $sporaParent $sporaOldRef $sporaOldRun runs/spora-gelu24-recurrent-layer19-v2 runs/spora-gelu24-recurrent-layer22-v1 --steps 32 --learning-rate .0005
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_placement $sporaOldRef $sporaNewRef --recoveries $sporaOldRun $sporaNewRun
```

固定 640 和 1,536 个输入词元复测首词延迟、16 次增量前向和实际保存的张量。每一种结构单独热身，交换条件顺序，三次取中位数。

```powershell
.\.venv\Scripts\python.exe -u -m scripts.benchmark_spora_recurrent_dialogue $sporaParent "$sporaOldRun/timing.json" "$sporaOldRun/layer-19" "$sporaOldRun/layer-22" --timing-only --extend --length 640 --length 1536
.\.venv\Scripts\python.exe -u -m scripts.benchmark_spora_recurrent_dialogue $sporaParent "$sporaNewRun/timing.json" "$sporaNewRun/layer-21" "$sporaNewRun/layer-24" --timing-only --extend --length 640 --length 1536
```

交叉组合复用已逐题对齐的父模型答复。载入时校验父模型权重、题集、个体和位置上限，每组省去重新生成 48 条父模型答复。

```powershell
.\.venv\Scripts\python.exe -u -m scripts.benchmark_spora_recurrent_dialogue $sporaParent runs/spora-recurrent-morph-19-24-extended-v1.json runs/spora-gelu24-recurrent-layer19-v2 runs/spora-gelu24-recurrent-layer24-morph-v1 --dialogue-only --extend --parent-reference $sporaOldRef
.\.venv\Scripts\python.exe -u -m scripts.benchmark_spora_recurrent_dialogue $sporaParent runs/spora-recurrent-morph-21-22-extended-v1.json runs/spora-gelu24-recurrent-layer21-morph-v1 runs/spora-gelu24-recurrent-layer22-v1 --dialogue-only --extend --parent-reference $sporaOldRef
```
