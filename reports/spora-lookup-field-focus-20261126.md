# 书目与用品同时出现时的字段定位

上一轮成组增加四类问法，没能提高未见措辞的整体地址识别。本轮用同一人物的书目和用品造出成对请求。每对使用同样的两项，只交换哪项要查、哪项暂缓，也交换分句顺序。用 48 条训练视图学习分句选择，24 条留出视图使用不同人物和三套新措辞。每套含书目与用品的交换及两个句序。字段头、意图头、阈值和循环骨干沿用上一轮的原组。StateWise 分别检验状态识别与执行，本轮也分别看分句、字段和实际提议。 [上轮实验](spora-switch-address-matched-20261126.md) · [StateWise](https://arxiv.org/abs/2610.33766) · [PassiveQA](https://arxiv.org/abs/2604.04565)

| 条件 | 首选字段正确 | 两侧字段均正确 | 正确地址提议 | 8 条暂缓或改写请求中的多余提议 |
| --- | ---: | ---: | ---: | ---: |
| 字段和意图都看整句 | 10/24 | 0/12 | 0/24 | 0 |
| 字段看模型所选分句，意图看整句 | 16/24 | 4/12 | 0/24 | 0 |
| 字段和意图都看模型所选分句 | 16/24 | 4/12 | 0/24 | 4 |
| 字段看人工标注分句，意图看整句 | 16/24 | 4/12 | 0/24 | — |

分句选择在这批留出题中答对 24/24。书目从 0/12 提到 4/12，用品从 10/12 提到 12/12。人工分句对照得到同样结果，说明剩余错误出现在字段头本身。整句意图头没有放行任何一条正例。改用分句意图后放行四条正例，但没有正确提议地址，并对四条负例提出了多余的地址。需要在整句层面区分查阅、暂缓和引用，再用局部语义识别用户所问的字段。

最终运行约 87 秒，输入权重摘要前后一致，126 项相关测试通过。最初新增负例时，评测假定每个分句都包含整句中的姓名，遇到无姓名分句后中止。修正为按姓名映射候选，无姓名分句不生成地址提议。紧接的一次复跑提前退出且未留下诊断，随后完整复跑通过。这组测试只检查候选和提议，没有读取持久记忆。样本来自两类事项、两位人物和三套留出措辞，权重留在实验目录。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_field_focus runs/spora-switch-address-matched-v1 runs/spora-lookup-field-focus-new --recurrent @weights
```

逐题结果与分句头权重位于 `runs/spora-lookup-field-focus-v4`。
