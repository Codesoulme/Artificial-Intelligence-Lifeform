# 真实中文字段监督向个人记忆寻址的迁移

上轮在每人有四项记录的档案中训练联合头，两种模型都没有发起正确读取。给定查阅许可后只排对 3/8。这轮用真实中文任务对话监督请求字段，再到同一份个人档案检验迁移。CrossWOZ 提供用户请求的领域和字段标注，CalOn 与 Xing USP 分别评估类别与实例的归属关系。 [上轮对照](spora-dense-joint-address-20261126.md) · [CrossWOZ](https://aclanthology.org/2020.tacl-1.19/) · [CalOn](https://arxiv.org/abs/2610.17113) · [Xing USP](https://arxiv.org/abs/2610.34343)

## 设置

使用 CrossWOZ 固定版的官方训练、验证和测试分区。抽样按同时请求多个字段、相邻请求切换领域、当前没有字段请求三类进行，选出 1,152 条训练、288 条验证和 288 条测试请求。输入只含当轮用户原话及紧邻的上一条系统回复。长用户句不入选，上一条系统回复保留末尾 120 字。词汇表固定为训练分区出现的 68 种「领域／字段」组合。每条抽中的标注重新和原始资料核对，三个分区删除完全相同的输入。

保存两层循环版的原有权重，提取第 12 层表示。三组以同样初始化、同样的个人档案训练顺序和同样的更新步数训练联合地址头。第一组只训练个人地址，另外两组给共享投影增加公开任务的多标签损失。一组使用正确的请求字段，一组打乱整条标签，保留标签频率和计算预算。公开任务与个人档案分别校准。个人档案保留 16 条真实记录，使用上轮的 12 条题以及另外 12 条新题，再分别模拟隐藏正确地址。

## 学习与迁移结果

三组各训练 256 步，个人地址使用 512 个训练视图。个人档案对照复现上轮局部联合头全部 13 个权重张量。公开题在官方验证上选输出阈值，个人地址沿用独立的 48 条校准题。

| 指标 | 仅个人地址 | 打乱公开标签 | 正确公开标签 |
| --- | ---: | ---: | ---: |
| 公开测试字段微平均 F1 | — | 10.5% | 50.6% |
| 公开测试字段集合完全正确 | — | 0/288 | 102/288 |
| 公开多字段请求集合完全正确 | — | 0/120 | 17/120 |
| 原个人档案题实际读对 | 0/8 | 0/8 | 0/8 |
| 新个人档案题实际读对 | 0/8 | 0/8 | 0/8 |
| 给定查阅许可后，原题联合选址 | 3/8 | 4/8 | 2/8 |
| 给定查阅许可后，新题联合选址 | 6/8 | 6/8 | 5/8 |

三组在两套个人档案题上的原始最高分均为弃权。移除正确地址后也全数弃权。联合头的档案读回实验没有发生实际错读。真实公开标签在公开测试中带来清晰收益，但共享投影这一接法没有让个人查阅启动。训练和评估约 150 秒。源档案、原模型和辅助语料摘要保持一致。62 项版次单元测试通过。逐题结果保存在 `runs/spora-crosswoz-field-transfer-v1/result.json`。

## 固定人物或事项的诊断

独立进程重载后，公开测试的全部汇总指标复现。使用相同阈值去掉上一条系统回复，正确标签组的公开微平均 F1 从 50.6% 降为 45.2%，多字段完全正确从 17/120 降为 6/120。去掉回复后，多报字段从 169 增至 502。上一轮的文本承接对这组公开任务有实测用途。

| 限制可选范围，八条正例 | 仅个人地址 | 打乱公开标签 | 正确公开标签 |
| --- | ---: | ---: | ---: |
| 原题，给定真实事项后选对人物 | 8/8 | 8/8 | 8/8 |
| 原题，给定真实人物后选对事项 | 3/8 | 4/8 | 4/8 |
| 新题，给定真实事项后选对人物 | 8/8 | 8/8 | 8/8 |
| 新题，给定真实人物后选对事项 | 8/8 | 8/8 | 7/8 |

这些对照只比较地址分数。各组在已知其中一项时能保住较多坐标，联合判断与是否发起查阅仍存在误差。从同一分数矩阵改取边际最大、先人物或先事项，留出题正确数也没有提高。下一轮单独训练「应当确认查阅」和「应当继续对话」，固定已有地址分数，观察是否能恢复实际查阅。随后再用局部证据监督联合地址，分别衡量两处改善。*Lost with a Map* 将识别结构、解析值与执行分开，适合沿用为诊断。 [Lost with a Map](https://arxiv.org/abs/2609.33883)

诊断含模型加载和两次公开测试前向，用时约 55 秒。结果在 `runs/spora-crosswoz-field-transfer-v1/components.json`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
$corpus = 'runs/research/crosswoz-df82c9f'
$manifest = 'runs/research/crosswoz-positive-fields-v1'
$profile = 'runs/spora-dense-joint-address-v1'
$out = 'runs/spora-crosswoz-field-transfer-new'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_crosswoz_field_transfer $corpus $manifest runs/research/xing-usp-883c7dc $profile $out --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_crosswoz_transfer_components $out $corpus $profile "$out/components.json" --recurrent @weights
```
