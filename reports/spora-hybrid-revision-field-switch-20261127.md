# 固定问法切换查阅字段

上一轮双循环版在新书标签题中有三次给正确值打了更高分，生成时却转向其他内容。本轮固定历史、答复格式和生成设置，分别询问同一对话中的更新字段和保留字段。用相同的父检查点比较完整注意力与第 19、22 层循环布局。五种场景各用两套取值和两种更正距离，每组包含 20 对提问、40 条答复。输入为 702–720 个词元。 [同提示生成实验](spora-hybrid-revision-generation-20261127.md) · [Lost with a Map](https://arxiv.org/abs/2609.33883)

| 精确匹配指标 | 完整注意力父检查点 | 第 19、22 层循环版 |
| --- | ---: | ---: |
| 更新字段，近距 | 6/10 | 6/10 |
| 更新字段，远距 | 5/10 | 5/10 |
| 保留字段，近距 | 7/10 | 7/10 |
| 保留字段，远距 | 9/10 | 9/10 |
| 同一历史的两个字段都答对 | 9/20 | 9/20 |
| 问更新字段时复述旧值 | 3/20 | 3/20 |

两组在 40 条答复中有 36 条逐字一致，其余四条都出现在图书场景。询问标签时，两组各答对 2/4，另一次都转而回答存放处。父检查点给出完整存放名称「蓝边矮柜」，循环版写成「蓝柜」。两组在询问存放处时也各答对 2/4。只按完整值做自动匹配，会把循环版这次答错字段漏计。

两组各有四次正确填写样张数量但省略「份」。接受该写法时，两组更新字段都为 15/20。全部答复正常结束。结果覆盖固定五种场景中的字段切换，图书用例显示两组都可能答到另一个字段。后续可以对调两个字段在历史中的位置及标签，分开测语义混淆和位置偏好。

运行约 256 秒，受检文件与个体状态摘要一致，35 项相关测试通过。逐题词元和输出位于 `runs/spora-hybrid-revision-field-switch-v1.json`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_recovery runs/spora-composition-late-gates-20-23-24-v1/candidate.pt runs/spora-hybrid-position-recovery-v1 runs/spora-hybrid-revision-field-switch-new.json --revision-field-switch --recall-transfer data/recurrent-layout-recall-transfer-20261126.json --layout '19,22'
```
