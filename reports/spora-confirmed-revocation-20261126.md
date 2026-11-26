# 已确认地址的撤回识别与独立版次判断

前轮的分句模型在追加和改查时都能选入新地址，但总是保留原地址。本轮冻结骨干和候选动作头，单独训练原项撤回、末问版次两个小头。撤回实验限定起始地址已确认、候选动作头建议选入新地址。撤回头只影响当前请求的范围，实验不操作持久档案。 [前轮逐项动作](spora-lookup-clause-trace-20261126.md) · [否定对照](https://arxiv.org/abs/2610.09571) · [StateWise](https://arxiv.org/abs/2610.33766)

撤回头使用 60 个训练视图，来自十对保留与撤回句式及三种前缀。校准另有六条句子，选择的置信阈值为 0.5。先前反复检查过的 48 对请求用来复现旧失败，只有六组开场及地址组合。专项头在这些组里全部保留追加侧的原地址、移出改查侧的原地址，配对通过由 0/48 改为 48/48。这里计的是状态更新，尚未按预测执行档案读取。

两套独立措辞各含六条保留和六条撤回。两套结果均为撤回 6/6、保留 3/6，合计 18/24。两套各只有三对完整通过。模型将带「别拿掉」的保留请求、引用他人撤回但本人要求继续查的请求判成撤回，部分置信度接近 1。校准集上六条都答对，未能暴露这些错误。候选继续保留为实验权重。

版次头单独训练于 32 条问法。旧对照的 12 条题已在上一轮分析过，用于回归。另写 12 条问法，包含首版、当前版、上一版、引用与改写，每类三条。两套各通过 12/12，含此前旧头答错的三条上一版问法。版次评测只测请求分类，覆盖的每类措辞数量有限。

两个小头保存并重载后完成评估。第二次运行约 93 秒，原配对及旧问法分数与首次运行一致。实验前后冻结骨干及动作头摘要一致。下轮需要优先测否定指向与说话人归属，并在保留类加入独立校准族。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_confirmed_revocation runs/spora-lookup-transitions-clauses-v1 runs/spora-confirmed-revocation-new --recurrent @weights
```

逐条结果与权重位于 `runs/spora-confirmed-revocation-v2`。
