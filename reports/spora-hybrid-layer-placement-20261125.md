# DeepSeek 孢子的双层注意力迁移对照

现有孢子把第 19、22 层改成了近期精确注意力加循环历史，并保留具名个体记忆。多次调循环写入和训练目标，在约 1,400 词元处仍难以同时回答两个有效字段。接下来比较替换层的位置，在整网运行中检验这些层相互影响的结果。 [现有双层实现](spora-recurrent-attention-layer19-20261124.md) · [长前文多字段对照](spora-recurrent-field-composition-20261124.md)

11 月 18 日的 FlashMorph2 研究发现，用原骨干的隐藏状态单独评价各层，以及用训练前的权重给替换位置排序，都会漏掉真实组合的误差与补偿。它用固定预算采样离散层配置，执行整网前向，同时按后续恢复目标调整权重。论文实验使用 Qwen3 和线性注意力变体。孢子将验证其中两项可迁移的做法，即直接运行真实层组合，以及按恢复后的表现评价组合。 [FlashMorph2](https://arxiv.org/abs/2611.24809) · [论文引用的线性注意力实现](https://github.com/fla-org/flash-linear-attention)

另有近期论文在若干混合推理模型中观察到，低精度的循环状态会增加推理长度和耗尽预算的次数。孢子的现有循环历史用 FP32 累加，本地组间保持这一精度，并对照近期键值所用的 FP16。当前开放中文推理试验用的是另一份封装检查点，因此不拿它的长答现象归因于循环状态精度。 [循环状态精度研究](https://arxiv.org/abs/2611.12127) · [孢子循环注意力源码](../spora/research_recurrent_attention.py)

## 固定替换预算

先用第 19、22 层的原始对齐产物，与第 21、24 层的同规格初始对齐产物比较。层号从零开始，两组都替换两层，间隔同为三层。共同父检查点是 `runs/spora-composition-late-gates-20-23-24-v1/candidate.pt`，保留第 5、17 层的持久读取路径。两组使用 8 词元块、相同中文材料、160 步局部混合对齐和 32 步整网预测对齐。核对权重元数据后，父模型前馈门控所在层为 21、24、26。两组使用同一副冻结的门控权重。

局部对齐只用来初始化。随后对两种真实组合使用相同的成对样本、更新数和生成条件，在各自的整网传递下恢复。比较恢复前后的长前文合并作答、有效与过期值、32 条受控持久读取、实际状态大小和延迟。现有组合充当对照，额外报告完整注意力父模型的可达到水平。该试验验证两种层组合，论文中的全网联合搜索属于后续研究范围。

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
