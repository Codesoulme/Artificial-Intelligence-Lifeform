# 同一提示下的候选评分与自由生成

上一轮的逐词元评分显示，两种布局都在 33/40 条单字段问题上给新值比旧值更高的分数。本轮复用保存的分数，使用相同的模型权重与 702–717 词元的提示做贪心生成，生成上限为 48 词元。逐题核对场景、提示长度和两个候选的词元，并检查生成器收到的提示词元与评分代码重新构造的词元逐项一致。测试仍覆盖五种场景、两套取值、新旧互换及远近两种距离。 [候选评分](spora-hybrid-revision-likelihood-20261127.md) · [Lost with a Map](https://arxiv.org/abs/2609.33883)

| 指标 | 完整注意力父检查点 | 第 19、22 层循环版 |
| --- | ---: | ---: |
| 候选评分偏向新值 | 33/40 | 33/40 |
| 自由生成精确匹配新值 | 24/40 | 22/40 |
| 候选评分偏向新值，但生成省略或漏掉它 | 9/33 | 11/33 |
| 其中只省略量词「份」 | 8 | 8 |
| 其中开头已偏离新值首词元 | 1 | 3 |
| 自由生成包含旧值 | 3/40 | 4/40 |

八次量词省略都发生在样张数量场景，两组都生成了正确的数字。循环版剩余三次偏离均发生在新书标签场景，其中两次转而回答未登记书籍的存放问题，另一次只复述了字段名。父检查点在这个场景也有一次答成存放问题。按本次逐题复查，接受正确数字省略量词时，两组的命中数分别为 32/40 和 30/40。这项放宽仅用于解释严格评分的差距。

自由生成和候选评分共享提示，候选评分测的是两个指定值之间的相对偏好，自由生成可输出其他词。两组各 40 条生成全部正常结束。这次使用的是从恢复实验加载的双循环权重，结果只覆盖单字段更正。下一步优先用同一格式改变待回答的字段，检验模型能否稳定区分标签和存放处，再判断是否需要为生成增加字段约束。

实验约 234 秒。受检文件和个体状态摘要一致，35 项相关测试通过。逐题分数、答复和生成词元位于 `runs/spora-hybrid-revision-generation-v1.json`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_recovery runs/spora-composition-late-gates-20-23-24-v1/candidate.pt runs/spora-hybrid-position-recovery-v1 runs/spora-hybrid-revision-generation-new.json --revision-likelihood --revision-likelihood-reference runs/spora-hybrid-revision-likelihood-v1.json --recall-transfer data/recurrent-layout-recall-transfer-20261126.json --layout '19,22'
```
