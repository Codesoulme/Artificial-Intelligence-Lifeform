# 固定注意力键后交换持久记忆值

上一轮增大读取权重会同时放大正确和错配档案的影响，新增编号题在张量路径均未答出。此次把第 23 层的档案拆成键和值两部分，单独测哪一部分能令答案跟随记录。*The Key Handoff* 用供体与接收端中对应不同答案的对照定位读取通路。MIB 则强调把因果干预与具体可观测变量对应。本轮从同批档案交叉取键和值，逐样本记录两种编号的相对分数和真实首词元。 [上一轮权重扫描](spora-memory-attention-mass-20261126.md) · [The Key Handoff](https://arxiv.org/abs/2609.32942) · [MIB](https://proceedings.mlr.press/v267/mueller25a.html)

## 对照

沿用五个场景、十道问题，每层读取档案末 22 个词元。每道问题在偏置 0 和 4 下分别比较四种组合。两种同档组合来自完整旧档案，两种跨档组合固定各自的键，换成同场景另一份编号分配的值。问题、读取位置和输出层固定。所有键值张量均取自上轮已存的文件，同场景两份登记的字数和词元位置对齐。

只修改 V 时先核对各查询头分给档案的权重逐项相同。完整档案条件还须逐条复现上一轮首选词元、两编号相对分数及注意力权重。比较首词元是否随不同值变为对应编号，并计算只换键、只换值时正确编号相对另一编号的分数差变化。编号只出现在档案，推理输入保持当前问题。

## 结果

完成 80 次读取，40 次完整档案读取与上一轮逐项一致。固定键的 40 组对照中，两个值获得的注意力权重在全部查询头上逐项一致。

| 偏置 | 只换值后答案两边均正确 | 只换键后答案两边均跟键对应编号 | 只换值时向目标移动的中位分数差 | 只换键时向目标移动的中位分数差 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1/20 | 0/20 | 0.554 | -0.020 |
| 4 | 3/20 | 1/20 | 1.910 | 0.063 |

在两种读取强度下，只换值都有 14/20 组向对应编号移动。偏置 4 时，物品交接地点使用任一固定键，都能随档案值输出对应编号。校刊寄送日期也有一组完整跟随。新增场景在每个偏置条件下仍为 0/8。该实验支持第 23 层的值确实传递了部分编号信息，也说明仅靠原生投影，读出不足以覆盖新问题。后续适配器在这层只修改持久值向量，以现有检索路径为对照。已有小型阅读适配器研究也在冻结语言骨干的条件下比较了不同的内部读出路径。 [冻结解码器的持久记忆适配](https://arxiv.org/abs/2603.22329)

独立读取约 63 秒。个体、模型、上一轮的档案和结果摘要一致。CPU 对照还验证了同一 Q 和 K 下交换 V 能改变计算值，同时保持每个头的权重。结果位于 `runs/spora-persistent-value-binding-v1/result.json`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaParent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaLayer19 = 'runs/spora-gelu24-recurrent-layer19-v2'
$sporaLayer22 = 'runs/spora-gelu24-recurrent-layer22-v1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_persistent_value_binding $sporaParent runs/spora-attention-mass-v1/bank.pt runs/spora-attention-mass-v1/result.json runs/spora-persistent-value-binding-new/result.json $sporaLayer19 $sporaLayer22
```
