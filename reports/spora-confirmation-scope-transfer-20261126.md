# 查阅确认头的分句顺序对照

前次将确认和寻址分开后，逐题分数显示，假设确认已放行，两批正例的地址各可排对 6/8。假设地址已知，确认头分别放行 3/8 和 4/8。此次先检验训练样本中的句末线索是否妨碍了该问则问。*Lost with a Map* 分开检验结构判断、值解析和执行。*ToolTurns* 检验澄清后实际完成的任务。两种方法用于本轮分解和回放。 [前轮结果](spora-read-confirmation-action-20261126.md) · [Lost with a Map](https://arxiv.org/abs/2609.33883) · [ToolTurns](https://arxiv.org/abs/2611.14925)

固定孢子骨干、第 19 层特征、联合地址头、16 条重启后的记忆档案和训练预算。对照重载前轮模型。候选用等量的 64 对新样本重训确认头，把需要查阅的内容放到开头、中间和结尾，改变无关请求和撤回出现的位置。独立校准集、阈值选择规则和公开负例分区相同。报告原先的最近一批留出题和新写的 12 条题，后者含八条应读取和四条无需查阅。新题与此前训练、校准和评测的去姓名文本均无完全重复。两批仍使用同一份档案。

| 指标 | 前轮头，原题 | 重训头，原题 | 前轮头，新题 | 重训头，新题 |
| --- | ---: | ---: | ---: | ---: |
| 应查题通过动作阈值 | 4/8 | 0/8 | 5/8 | 4/8 |
| 动作已知时地址排对 | 6/8 | 6/8 | 5/8 | 5/8 |
| 经确认实际读对编号和值 | 3/8 | 0/8 | 3/8 | 1/8 |
| 提议错误地址或在无需查阅时提议 | 1 | 0 | 3 | 3 |
| 用户确认前的读取次数 | 0 | 0 | 0 | 0 |

前轮结果逐项复现。新训练组在相同校准规则下选出 0.90 阈值，前轮为 0.40。两组均跳过 128 条公开普通请求。新题把两项模型判断都换成正确输入时，全部八条在确认后读到了预期编号及值。耗时约 69 秒。62 项版次测试通过，档案与原权重摘要一致。

这批改动没有提升正确读回。样本措辞和分句位置同时改变，结果无法单独归因于分句位置。新题也显示寻址与动作均有失败，单独更换其中一项的上限各为 5/8。保留前轮头作为对照。近期的混合模型分析表明，循环层形成的查询与后续注意力取值有可分工的通路。后续对比头部在循环替换前后的表示，检验结构是否改变了可用的请求信息。 [Who Asks, Who Answers](https://arxiv.org/abs/2611.01909)

## 复现

逐题分数、回放和候选权重位于 `runs/spora-confirmation-scope-transfer-v1/`。

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_confirmation_scope_transfer runs/spora-read-confirmation-action-v1 runs/spora-crosswoz-field-transfer-v1 runs/spora-dense-joint-address-v1 runs/research/xing-usp-883c7dc runs/spora-confirmation-scope-transfer-new --recurrent @weights
```
