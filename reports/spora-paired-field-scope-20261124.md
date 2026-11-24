# 用成对问句学习孢子的字段读取范围

## 方法

上一轮已经能把正确的值绑定到相应角色，但会把排除项误判成必需或可选。本轮冻结原字段选择、角色展开、观察相关性、语言骨干和个体，只增加一个联合范围判定。它在保留原计划、只读第一项、只读第二项、第一项必需且第二项可选、反向可选以及跳过读取之间选择。

每个训练对使用同一格式，只让两个字段交换必需、排除或可选的位置。编码时也交换状态字段的描述顺序，将两个方向的输出还原到相同字段后再平均。成对损失要求每一侧对正确方案的分数高于相反方案。题目的特征输入含字段名称，槽位的实际值不参与范围分类。给出范围头的独立校准按六类中表现最差的一类优先选门槛。置信不足或预测保留原计划时，回到原字段选择。明确请求读出的新计划也需匹配当前主题。

真实生成包括此前两批完整与缺项记录、旧的排除与可选措辞、普通题，以及六组固定状态的成对新题。每组交换的只是请求字段，事实记录完全相同。成对新题沿用已知字段名，旧措辞组还包含别名和不直接复述字段名的写法。用于范围学习的讨论参考局部决策的成对诊断和逐项指令评测。 [上轮角色绑定](spora-role-bound-realization-20261124.md) · [Contrast Sets](https://arxiv.org/abs/2004.02709) · [FollowBench](https://arxiv.org/abs/2310.20410)

## 结果

训练使用 624 条问题，其中 168 对只交换角色，参数更新 220 次。独立校准选择 0.4 的置信门槛。按保留原计划、只读第一项、只读第二项、两个可选方向和跳过顺序，校准正确数为 12、10、5、0、0、12，各类均 12 条。校准用的可选措辞仍超出这套训练问法的泛化范围。

| 真实生成对照 | 原字段计划 | 成对范围判定 |
| --- | ---: | ---: |
| 固定状态换问法，两侧都按要求切换 | 0/6 | 4/6 |
| 固定状态问句，完整字段计划 | 4/12 | 9/12 |
| 固定状态问句，出现所要求省略的值 | 5/12 | 3/12 |
| 此前两批排除题，出现所要求省略的值 | 5/12 | 3/12 |
| 此前两批可选题，可选值缺失时仍完整回答必需值 | 5/12 | 9/12 |
| 明确排除措辞的普通题，复现空状态答复 | 5/6 | 6/6 |

固定记录的两条库存问句仍把至少一个排除项当成可选。花圃一题把应给出和应排除的字段颠倒。这三条都计为失败。已评估的 48 条完整取值答复和 48 条缺一项答复与上轮逐条相同，完整题保持在 9/12 和 8/12 双向通过，缺项分别为 18/24 和 12/24 的角色绑定通过。32 条持久读取与上轮逐条相同，个体摘要一致。34 项相关测试通过。

训练和全面生成验证约 390 秒。独立进程重载该范围头，逐条复现校准门槛、字段计划、真实答复和持久读取。逐题记录位于运行目录 spora-paired-scope-v1。当前实验使用已知主题的字段名与给定记录。后续应扩大同一个状态下的方向冲突、别名、缺字段与多于两项的对照，再与真实观察的角色推断及写入连接。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$sporaBase = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaSplit = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
$sporaAction = 'runs/spora-span-actions-v1'
$sporaGate = 'runs/spora-observation-read-gate-v1'
$sporaTyped = 'runs/spora-typed-fields-v1'
$sporaOutput = 'runs/spora-paired-scope-v1'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_paired_field_scope $sporaBase $sporaAction $sporaGate $sporaTyped $sporaOutput "$sporaSplit/layer-19" "$sporaSplit/layer-22"
.\.venv\Scripts\python.exe -u -m scripts.train_spora_paired_field_scope $sporaBase $sporaAction $sporaGate $sporaTyped $sporaOutput "$sporaSplit/layer-19" "$sporaSplit/layer-22" --verify
```
