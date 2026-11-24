# 长前文中的字段读取与合并答复

沿用六种新场景及各自两套取值。从同一份前文分别问更正后的值、保留的另一个值。两次只读调用各自独立，第一问的回复不会进入第二问。用合并提问的答复作对照。单字段判分要求包含该字段、没有旧值并正常结束。逐字检查不评价额外内容的准确性，也不能证明模型在合并提问时读取了相同的信息。 [长前文的多项检索评测](https://arxiv.org/abs/2404.06654)

| 模型与前文 | 单问新值 | 单问保留字段 | 分别问两项都答对 | 合并问完整答对 | 两次单问全对且合并失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整注意力，短前文 | 10/12 | 11/12 | 10/12 | 7/12 | 4/12 |
| 完整注意力，约 1,400 词元 | 7/12 | 9/12 | 5/12 | 1/12 | 4/12 |
| 两层循环注意力，约 1,400 词元 | 7/12 | 12/12 | 7/12 | 1/12 | 6/12 |

最长的库存题合并问只答「九只」，分开问识别出了「九只」和接收人。节目单一题合并问只答领唱人，单问曲名则答出了「晚风归舟」。另有花圃题在合并回答时使用了旧工具。长前文也降低了父模型分别答对两项的比例。前文长度、干扰文字以及事实和问题的距离同时变化，本轮无法区分三者。 [长前文中的干扰与位置效应](https://arxiv.org/abs/2307.03172)

逐题记录位于 `runs/spora-recurrent-split_gated_block_delta-warm-v1/field-composition.json`。评测重用同一摘要的长前文合并回答，短前文的合并回答和所有单问回答由新进程生成。

## 按字段均衡训练

从同一循环检查点启动两组。每组 24 次更新，抽样器、学习率、更正词元分差损失及完整回答监督相同。实验组额外对正确答案的日期、地点、数量等字段分别计算平均损失，再对字段等权平均。把该项与原完整回答损失等权混合，仍监督标签、标点及结束词元。字段边界来自已有训练模板的标注。48 对训练样本经实际分词核对，两个条件的输入和目标词元相同，96 个答案的三个字段均有对应监督位置。留出对照集中在完整答复、旧值、保留字段和受控持久读取。

| 评测 | 起点 | 原完整答案损失 | 增加字段均衡 |
| --- | ---: | ---: | ---: |
| 约 700 词元换值题，单题 / 双边都对 | 6/10 · 2/5 | 5/10 · 1/5 | 7/10 · 2/5 |
| 约 1,400 词元换值题，单题 / 双边都对 | 2/10 · 0/5 | 1/10 · 0/5 | 1/10 · 0/5 |
| 原更正题，约 700 / 1,400 词元 | 5/10 · 4/10 | 7/10 · 3/10 | 7/10 · 3/10 |
| 六类新场景，约 1,400 词元完整通过 | 1/12 | 1/12 | 1/12 |
| 新场景，新值 / 旧值 / 保留字段出现 | 4/12 · 2/12 · 5/12 | 4/12 · 2/12 · 4/12 | 4/12 · 3/12 · 4/12 |
| 开放题规则通过，整组 / 逐轮 | 2/6 · 9/18 | 2/6 · 10/18 | 2/6 · 10/18 |
| 受控持久读取 | 32/32 | 32/32 | 32/32 |

字段均衡只在约 700 词元的换值题上比同预算对照多答对两题。两组完整通过的新场景是同一道取件题，持久读取逐题一致。两组的训练峰值 CUDA 分配均约 4.14 GiB。保留续训前的循环权重作为下一轮实验基线。之后应冻结循环状态的读写，单独测试普通生成的读出路径是否限制多个字段在同一答复中出现。两个续训结果位于 `runs/spora-recurrent-field-coverage-0-v1/` 和 `runs/spora-recurrent-field-coverage-1-v1/`。

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$sporaCandidate = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
$sporaParent = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
.\.venv\Scripts\python.exe -m scripts.audit_spora_recurrent_field_composition $sporaParent "$sporaCandidate/field-composition.json" "$sporaCandidate/layer-19" "$sporaCandidate/layer-22" --parent-report "$sporaCandidate/far-transfer-parent.json" --recurrent-report "$sporaCandidate/far-transfer.json"
foreach ($sporaFieldWeight in '1','0') {
    $sporaOutput = "runs/spora-recurrent-field-coverage-$sporaFieldWeight-v1"
    .\.venv\Scripts\python.exe -m scripts.train_spora_recurrent_counterfactual $sporaParent "$sporaCandidate/far-transfer.json" $sporaOutput "$sporaCandidate/layer-19" "$sporaCandidate/layer-22" --steps 24 --field-weight $sporaFieldWeight --transfer data/recurrent-phase-transfer-v1.json
    if ($LASTEXITCODE) { break }
}
```
