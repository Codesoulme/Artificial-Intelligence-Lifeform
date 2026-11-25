# 自然请求的读取意图对照

对「沿用已登记内容」和「另选方案」的请求，字段相似度不能单独决定是否读取。本轮冻结逐字段选择结果，单独训练二分类头判断用户是否要求使用已有记录。使用原来 156 条训练请求，再加入 24 对同一主题的反事实措辞，共 204 条。用冻结 DeepSeek 的前 12 层提取整句特征，训练 12 轮，固定概率阈值 0.5。已有显式版次请求保留原解析决策。 [逐字段对照](spora-field-evidence-20261125.md) · [域外意图分类](https://arxiv.org/abs/1909.02027) · [Hy-MultiTurn](https://arxiv.org/abs/2607.29196)

| 已有评测集 | 原决策完整读对 | 加读取判断后 | 原决策误读 | 加读取判断后 |
| --- | ---: | ---: | ---: | ---: |
| 旧组 | 7/13 | 7/13 | 4/10 | 3/10 |
| 新措辞组 | 3/11 | 2/11 | 1/9 | 0/9 |

旧组仍有三次读取了用户未要求的记录。新措辞组的五次错读或部分读取均保留，因为新判断按整句决定读取，无法判断具体哪项要排除。结果基于 43 条已存的寻址记录，本轮没有再次执行张量读取或生成，也没有在新的独立评测集上选择阈值。独立进程复现了全部 43 条概率与判定。候选保留在研究目录。下一步优先把用户对单个字段的肯定、排除和未指明分别标注，评估时同时统计多读、少读及选对的完整集合。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$sporaBaseline = 'runs/spora-field-evidence-v1/result.json'
$sporaRun = 'runs/spora-read-authority-new'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_read_authority $sporaBaseline $sporaRun
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_read_authority $sporaBaseline $sporaRun --verify
```
