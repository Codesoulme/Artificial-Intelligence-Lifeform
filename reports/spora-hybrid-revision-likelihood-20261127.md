# 交换新旧值后的逐词元评分

上一轮在双字段 A/B 题里，两组都偏向字母 B。此次沿用五种场景及各两套取值，将问题固定为只填更新字段。分别把两个候选值接到同一提示后，计算完整候选的逐词元条件对数概率，同时记录均值和总和。每个场景还交换一次新旧值，保留字段、干扰对话和提问保持一致。近距与远距使用相同材料和词元数，输入为 702–717 个词元。 [StateMemBench](https://arxiv.org/abs/2608.19652) · [上一轮](spora-hybrid-revision-stages-20261127.md)

| 指标 | 父检查点，近距 | 双循环版，近距 | 父检查点，远距 | 双循环版，远距 |
| --- | ---: | ---: | ---: | ---: |
| 按词元均值，新值分数更高 | 17/20 | 17/20 | 16/20 | 16/20 |
| 按词元总和，新值分数更高 | 17/20 | 16/20 | 16/20 | 17/20 |
| 按词元均值，互换前后均选中新值 | 8/10 | 7/10 | 7/10 | 7/10 |

两组在单字段候选评分上大多偏向当前值，汇总命中率相同，逐题仍有差异。此前 A/B 题还改变了问题形式和回答字段数，两轮差异无法单独归因于字母偏好。此次从两个已知候选中计分，开放生成仍需另外验证。下一步可统一单双字段的问法，将同一道题的候选分数和实际生成逐题配对，找出选中新值却未写入答案的情况。

每组共 80 次候选前向，两个布局同进程运行约 424 秒。父检查点、恢复权重、评测来源和个体状态摘要检查通过，35 项相关测试通过。逐词元结果位于 `runs/spora-hybrid-revision-likelihood-v1.json`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_recovery runs/spora-composition-late-gates-20-23-24-v1/candidate.pt runs/spora-hybrid-position-recovery-v1 runs/spora-hybrid-revision-likelihood-new.json --revision-likelihood --recall-transfer data/recurrent-layout-recall-transfer-20261126.json --layout '19,22'
```
