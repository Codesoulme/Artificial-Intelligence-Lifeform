# 分开训练查阅确认与地址选择

上轮给定真实人物或事项时，联合地址头能选对相当一部分坐标，三组实际测试却都未发起查阅。此次固定先前只用个人地址训练的参数，另训一个判断是否需要提出确认的小头。ToolTurns 从实际提问与补答的序列评分，PassiveQA 分开监督问询和普通答复。 [上轮结果](spora-crosswoz-field-transfer-20261126.md) · [ToolTurns](https://arxiv.org/abs/2611.14925) · [PassiveQA](https://arxiv.org/abs/2604.04565)

## 设置

每对人工样本共享人物和大部分措辞。正例继续询问旧安排，负例撤回、改为自行联系，或要求编辑原话。用整句平均和最后一个分句的平均作为两路输入，按角色将姓名替换成「甲」「乙」。比较第 12、19 层，在第一处循环注意力之前结束特征提取。两层使用同样初始化、样本和优化步骤。另取公开普通问句，训练和校准分开，公开测试分区只用于复核误触发。

个人档案沿用已重启的 16 条记录。两批各含八条口语查阅和四条其余请求。每条使用原入口的 `VersionChat` 回放，并把评测的动作和地址分数传给其既有口语查阅回调。第一轮只核对提议。正确提议收到「是，当前版」，错误提议收到「不对」。第二轮计实际记录编号和读回的值。该流程保留了入口的取消、编辑、提议与确认逻辑，并将模型打分固定在可观察的条件下。

## 结果

使用 64 对训练句和 12 对校准句。两组在 128 条公开普通请求上均没有触发。第 12 层模型两批各提议一个错误地址，未读对。第 19 层模型两批各提议一个错误地址，分别读对 2/8 和 3/8。两种深度均没有在确认前取值，用户拒绝错误提议后也没有取值。

| 第 19 层逐题分解 | 前批 | 新批 |
| --- | ---: | ---: |
| 动作放行且地址排对 | 2/8 | 3/8 |
| 假设动作已放行，地址排对 | 6/8 | 6/8 |
| 假设地址已知，动作放行 | 3/8 | 4/8 |
| 两项均正确时，经确认读对实际值 | 2/2 | 3/3 |

逐题分数表明，漏问盖过了后续地址选择，仍有两条正例在放行后会排错地址。两项均正确时，确认流程在这五条题上均能读回正确值。实验耗时约 80 秒，源档案和绑定权重摘要一致。逐条结果及两份头部权重在 `runs/spora-read-confirmation-action-v1/`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_read_confirmation_action runs/spora-crosswoz-field-transfer-v1 runs/spora-dense-joint-address-v1 runs/research/xing-usp-883c7dc runs/spora-read-confirmation-action-new --recurrent @weights
```
