# 第 24 层的定向蒸馏与 FP16 梯度下溢

第 19、22、24 层循环版加入整段隐藏状态监督后，中文生成与仅监督答复的对照逐条相同。本轮分别测整段误差、答复损失和定向误差传到第 24 层循环核的梯度。定向误差各用一半权重对齐提问末尾 16 个位置和预测事实字段的隐藏状态，仅用训练样本估计系数。两组沿用相同的初始化、抽样顺序、16 步预算与学习率，均处理 32,113 个输入词元。 [整段蒸馏对照](spora-hybrid-hidden-recovery-20261126.md) · [GradNorm](https://arxiv.org/abs/1711.02257)

未缩放时，8 条最长训练样本的整段梯度中位范数为 0.0000156，答复梯度的约 0.0032%。定向梯度的权重取答复梯度与定向梯度范数之比的中位数 9.57。定向梯度与答复梯度的余弦相似度从 -0.45 到 0.67，这个权重只平衡量级。

| 训练目标 | 第 24 层末端相对均方误差 | CrossWOZ NLL | 短更正 | 约 700 词元 | 约 1,400 词元 | 开放题逐轮 | 具名读取 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 答复监督 | 0.06698 | 4.97378 | 8/10 | 5/10 | 1/10 | 3/18 | 32/32 |
| 加整段对齐 | 0.06698 | 4.97408 | 8/10 | 5/10 | 1/10 | 3/18 | 32/32 |
| 加定向对齐 | 0.06338 | 4.96743 | 8/10 | 6/10 | 1/10 | 3/18 | 32/32 |

定向对齐让留出题第 24 层末端误差约降 5.4%，补回一条约 700 词元的发货更正题。三组在 CrossWOZ 48 段服务对话的首选词元准确率均为 27.53%。约 1,400 词元的更正及开放对话仍比当前双层基线低。定向训练与评估约 380 秒，CUDA 张量训练峰值约 4.08 GiB，源文件及个体状态摘要一致。 [双层基线生成](spora-hybrid-language-recovery-20261126.md)

另用相同 8 条样本对比反向传播倍率 1 和 256。第 24 层 FP16 激活上，整段损失的非零梯度元素平均占比由 0.0133% 升至 96.42%。折回原尺度后，整段梯度范数中位数为 0.00591，各样本比未缩放大 314 至 453 倍，仍只相当于答复梯度的中位 1.15%。结果说明原蒸馏的微弱更新部分来自下溢，也支持单独重跑整段训练，测量修正下溢后能恢复多少生成表现。 [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sources = @('runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1', 'runs/spora-gelu24-recurrent-layer24-morph-v1')
$checkpoint = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$test = 'runs/research/crosswoz-df82c9f/test.json.zip'
$reference = 'runs/spora-hybrid-hidden-recovery-v1/result.json'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_hybrid_hidden_recovery $checkpoint $test runs/spora-hybrid-targeted-recovery-v1 @sources --reference $reference
.\.venv\Scripts\python.exe -u -m scripts.train_spora_hybrid_hidden_recovery $checkpoint $test runs/spora-hybrid-loss-scaling-v1 @sources --reference $reference --check-loss-scaling
```
