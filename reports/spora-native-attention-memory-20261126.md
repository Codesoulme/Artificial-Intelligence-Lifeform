# 从循环层查询到原生注意力读取

上一轮交换孢子第 19、22 层末位置的完整隐藏向量，发现问哪个字段和读到哪个值可以部分分离。本轮区分后续精确注意力中的查询向量与其他信息路径。新查阅的 *The Key Handoff* 在多类混合模型上观察到循环层传递查询键、后续注意力根据它取值。*Attention Amnesia* 则报告，训练修改查询与键的投影可能损害长距离读取。本机沿用冻结权重，直接干预运行中的投影。 [上轮整向量交换](spora-recurrent-query-binding-20261126.md) · [The Key Handoff](https://arxiv.org/abs/2609.32942) · [Attention Amnesia](https://aclanthology.org/2026.emnlp-main.816/)

## 只交换查询向量

在第 20、23 层的融合投影中，只交换最末位置的 Q 分量。K、V、较早位置和进入注意力层的残差取自接收端。位置一致，替换发生在旋转位置编码之前。和整向量对照使用同一批场景、父检查点和循环权重，基线分数和首选词元逐条一致。表内只比较两版基线四种组合均答对的场景。分配相反时，按供体自身记录作答与按接收端记录选择另一个字段，会得到相反的编号。

| 前文 | 查询所在层 | 父版同分配 | 父版相反分配 | 循环版同分配 | 循环版相反分配 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 54–58 词元 | 20 | 2/12 | 3/12 | 3/12 | 3/12 |
| 54–58 词元 | 23 | 1/12 | 1/12 | 4/12 | 5/12 |
| 630–634 词元 | 20 | 1/4 | 1/4 | 1/4 | 1/4 |
| 630–634 词元 | 23 | 1/4 | 1/4 | 3/4 | 3/4 |

短题循环版第 23 层在相反分配下，目标编号相对另一编号的对数优势中位提升 11.29，父版是 3.86。整条隐藏向量交换在第 22 层仍有 9/12 成功，单独交换下一层查询时有 5/12。第 23 层是优先检验的读取位置，其贡献只覆盖部分样本。长题共有一组四种组合都正确，证据仍限于受控字段题。短题和长题分别约 71、211 秒，共 24 次自交换的完整词表分数一致。CPU 检查确认修改只落在各组查询的末位置，原整向量模式仍保留。逐题数据在 `runs/spora-recurrent-query-only-v1/`。

## 保存单层键值并在新进程读取

仓库早期的记忆注意力分支使用独立 softmax，参数有更新，但关闭分支后的生成保持一致。本轮参考 *Memory Inception* 的选择性键值分配，让持久键与当前输入的键在一层共享同一 softmax。*KV-LTM* 对全会话保存键值的成本给出了估算，本地试验将每份记录限制为单层 40 个位置。 [早期并行分支实验](pretrained-memory-attention-pilot-20261113.md) · [Memory Inception](https://arxiv.org/abs/2605.06225) · [KV-LTM](https://arxiv.org/abs/2611.21998)

写入进程分别从第 20、23 层读取六份短登记的原生键值张量，每场景保存两种相反分配。档案仅含投影张量、来源校验和场景索引。读出在独立进程进行，模型输入只有当前问题和相同的答复起始符，所有条件共享为记忆预留的位置。档案键值只进入选定层最后一个词元的注意力计算。用同一句问题在空状态、正常分配、交换两编号分配下运行，另保留登记可见时的结果作为参照。评测显式指定场景，实际读取的第一词元由原生输出层预测，供体键值来自冻结模型。当步读取及存档都属于实验入口。

| 读取条件 | 在两个编号间选对 | 首选词元是正确编号 | 同题两份档案均给出正确首词元 |
| --- | ---: | ---: | ---: |
| 空状态 | 6/12 | 2/12 | 0/6 |
| 第 20 层加载档案 | 6/12 | 1/12 | 0/6 |
| 第 23 层加载档案 | 8/12 | 4/12 | 1/6 |
| 登记置于可见前文 | 12/12 | 12/12 | 6/6 |

物品交接场景中，固定询问地点，切换第 23 层存档使首选词元由 `1` 变为 `2`，两次分别符合各自登记。展板领取的房间题也能让两个编号的相对次序随存档变动，不过完整词表的首选词元始终是左方括号。读取仍缺少普遍准确性，尤其校刊寄送两题的正确性没有因切换存档改善。

档案大小 290,830 字节，包含两处层位分别对应六份记录的键和值，每份 22–24 个词元。独立读取约 108 秒，源模型、孢子个体和档案摘要前后一致。CPU 检查核实旧键和当前键共用 softmax、分组查询匹配原注意力、布尔和浮点掩码一致。10 项循环注意力单元测试通过。首轮写入在核对摘要时将路径误传为字符串，修正后重新写入并完成跨进程读取。数据在 `runs/spora-persistent-attention-slots-v1/`。

下一轮先测第 23 层读到记忆的权重有多大，以及权重增加后是否仍区分正确和错误记录。保留位置、权重和问题的一致对照，再用不同措辞与训练未见值比较单层、两层和空档案。如果可靠取值能随读取范围提高，再把小型读出投影接到现有版本化快权重，考察新旧版次分别改动、普通答复保持及实际多词元生成。近期的参数记忆综述区分参数存放位置和运行中写入，冻结 Engram 的迁移实验也单独衡量了读取器的对齐作用。 [参数记忆综述](https://arxiv.org/abs/2610.08630) · [冻结记忆与读取器对照](https://arxiv.org/abs/2608.17050)

## 用本机权重复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaParent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaLayer19 = 'runs/spora-gelu24-recurrent-layer19-v2'
$sporaLayer22 = 'runs/spora-gelu24-recurrent-layer22-v1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_recurrent_query_binding $sporaParent runs/spora-recurrent-query-only-new/short.json $sporaLayer19 $sporaLayer22 --query-only
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_recurrent_query_binding $sporaParent runs/spora-recurrent-query-only-new/long.json $sporaLayer19 $sporaLayer22 --query-only --long
```

键值写入和读取用两个独立的进程执行。

```powershell
$sporaBank = 'runs/spora-persistent-attention-slots-new/bank.pt'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_persistent_attention_slots $sporaParent $sporaBank $sporaLayer19 $sporaLayer22
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_persistent_attention_slots $sporaParent $sporaBank $sporaLayer19 $sporaLayer22 --output runs/spora-persistent-attention-slots-new/result.json
```
