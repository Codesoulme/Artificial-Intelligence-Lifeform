# 口语问法的记忆查阅决策

上轮通过 Windows 入口测试六组问题。明确给定姓名、事项和版次时，两种入口都读对 6/6。换成口语后没有调用档案，也没有发起追问。本轮先给定正确地址，检验模型能否直接用原句读出名称，再训练一个同时预测查阅动作和地址的小型读出头。 [上轮回放](spora-recurrent-chinese-20261126.md)

TransMem 将事实与前后更正关系分开，HabitusMem 根据当前请求决定是否使用历史偏好。孢子已有可独立查阅的版次和张量读出，本轮把这两点落实为动作、人物、事项以及前后版次的共同预测。读取许可研究指出，答对文字与实际打开了哪些记录应分别核验。 [TransMem](https://arxiv.org/abs/2611.10683) · [HabitusMem](https://arxiv.org/abs/2611.25855) · [读取许可验证](https://arxiv.org/abs/2610.07309)

## 给定正确地址

使用两份已重启过的隔离档案，沿用原六组用例。三个条件都只给记忆读出器一条正确记录。用明确问法、保留口语原句并要求复制该记录、保留口语原句并取消强制复制。回答中应只出现所选的登记值，同时验证生成复制的位置。问句本身没有存入的具体值，生成前没有带入前轮文本。两份档案在测试前后摘要一致。

| 条件 | 默认版 | 两层循环版 |
| --- | ---: | ---: |
| 明确问法，要求复制 | 6/6 | 6/6 |
| 口语问法，要求复制 | 6/6 | 6/6 |
| 口语问法，自由决定是否复制 | 6/6 | 6/6 |

例如问「陶依萍换饮料以前，选的是什么？」时，循环版在给定上一版地址后自行复制「麦芽茶。」。这批题的读出通路具备处理原问法的能力。此次只测短事实，且地址由评测提供。用时约 48 秒，逐题结果在 `runs/spora-natural-lookup-cli-v1/oracle.json`。

## 学习动作和地址

在两层循环版上冻结骨干，从第 12 层提取特征。已登记姓名按原句中首次出现的次序映射为「甲」和「乙」，小头分别预测继续聊天、读取或追问，选择人物位置、四类事项及当前版、上一版、首版。训练数据把查阅、转述和撤回放在同组，同一人的字段词与时间词保持一致。文本中有两个姓名时，目标有时在前、有时在后。对缺失姓名、事项或版次的样本标注追问。只有要求查阅的样本训练地址预测。去除姓名替换后相同的例子，得到 305 条自编训练请求。

校准使用 81 条不同句式，共 24 条查阅、48 条非查阅和 9 条需澄清。候选阈值先比较错误查阅数，再比较完整查阅数。将校准后的头用于六组入口回放及另外五条追问与泛聊示例。预测地址经已有版次规则验证，查到一条实际记录后运行同一读出头。追问在此阶段计为动作预测，未进入真实的补答流程。两种提取方式分别取固定提问后缀的末位置，以及对原问句各字符对应的特征取平均。训练数据、头结构和优化参数相同。

| 指标 | 末位置原始 | 末位置校准后 | 原问句平均原始 | 原问句平均校准后 |
| --- | ---: | ---: | ---: | ---: |
| 六条口语题，选对记录并读出值 | 0/6 | 0/6 | 3/6 | 2/6 |
| 17 条受测题中的实际错读 | 1 | 0 | 2 | 0 |
| 该追问的样本，预测追问 | 2/3 | 3/3 | 2/3 | 3/3 |
| 其余 14 条中预测追问 | 0 | 1 | 2 | 5 |

原问句平均在校准后读回「麦芽茶」和「山城信」，两条分别涉及上一版和首版。未经校准时还读对一次用品，却在措辞相近的润色请求中读出了同一记录。阈值拦下了两次错读。校准后共有八条预测为追问，其中两条本可直接查阅，三条本应继续普通聊天。校准集里，两个头在阈值启用后都未完成任何一条查阅。校准前，末位置选全了 1/24，原问句平均选全了 6/24。这些结果支持保留原句更多信息作为下一轮基线，候选仍停留在隔离实验。第 12 层提供决策特征，第 19、22 层参与查阅后的生成，本轮差异对应特征的汇总方式。

末位置组重跑得到相同权重摘要及全部留出结果，分别用时 38 秒和 38 秒。原问句平均组用时 41 秒，权重文件约 0.6 MB。实验前后核对源档案、读出头、父检查点与两处循环权重摘要。59 项版次单元测试通过。逐题结果在 `runs/spora-spoken-query-plan-last-v2/result.json` 和 `runs/spora-spoken-query-plan-span-v1/result.json`。

下一轮先补监督数据。Xing USP 分别衡量动作、事项和实例归属，公开的中文 SMP2019、SMP2020 和 CAIS 涵盖交通、作品、设备等实际请求类别，适合作为「用户在提问，但无需读私人档案」的干扰。仍需专门编写带前后版次和不同人物的查阅正例。对新句式先按动作、人物绑定、事项和时间关系分别验收，再用 ToolTurns 的逐轮安排加入补答和改口项。 [Xing USP](https://arxiv.org/abs/2610.34343) · [公开中文语料目录](https://huggingface.co/datasets/XingChen-AGI/Xing4.0-29B-Intent-oss/tree/883c7dc7f52d380b6d4d80067a267ee6921cad39/General/train/dataset) · [ToolTurns](https://arxiv.org/abs/2611.14925)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$prior = 'runs/spora-natural-lookup-cli-v1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_natural_lookup_oracle $prior "$prior/oracle-new.json" --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_spoken_query_plan $prior runs/spora-spoken-query-plan-last-new --features last --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_spoken_query_plan $prior runs/spora-spoken-query-plan-span-new --features user-span --recurrent @weights
```
