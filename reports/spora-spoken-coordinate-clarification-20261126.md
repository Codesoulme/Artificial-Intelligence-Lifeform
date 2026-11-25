# 口语查阅中只追问缺失的人物或事项

前轮的动作头只能继续聊天或提议一个完整地址。本轮把确认完整地址、询问姓名、询问事项和普通聊天分开。ClariEdge 使用能从前文消解的难负例来评估是否需要追问。DiscoBench 和 ToolTurns 将识别疑义、提问及用户补答后的任务完成分开评分。这里以补答后读到的孢子张量记录及其实际值为准。 [前轮结果](spora-confirmation-recurrent-features-20261126.md) · [ClariEdge](https://arxiv.org/abs/2610.28686) · [DiscoBench](https://arxiv.org/abs/2606.27669) · [ToolTurns](https://arxiv.org/abs/2611.14925)

固定本机循环版、联合地址头、16 条已重启的档案和语言读出头。用第一处循环替换之前的冻结表示训练新头。监督包含 156 条自编中文请求和 96 条公开普通请求，校准使用 26 条单独撰写的请求及 32 条公开普通请求。自编训练、校准和留出在去姓名后没有重复文本。另留 128 条公开普通请求复核触发率。现有实验入口只接收已登记人物或事项构成的部分地址。用户补齐缺项和版次后才读取。

## 回放

21 条手写留出包含 4 条缺人、4 条缺事项、5 条地址完整或可由上轮明确查阅承接的请求，以及 8 条普通、撤回或润色请求。完整地址比较包含一条由实际前文消解的「那当前版呢」。错误提议在回放中收到取消，正确提议收到确认或直接补上缺项与版次。给定答案的对照由脚本把标注动作和已知坐标交给同一个聊天入口。

| 指标 | 明示规则 | 整地址提议 | 定向追问 | 按题目给定动作和坐标 |
| --- | ---: | ---: | ---: | ---: |
| 缺人，补答后读对 | 0/4 | 0/4 | 2/4 | 4/4 |
| 缺事项，补答后读对 | 0/4 | 0/4 | 1/4 | 4/4 |
| 地址完整或可由前文消解，读对 | 1/5 | 3/5 | 2/5 | 5/5 |
| 应查题提议错误 | 0 | 1 | 1 | 0 |
| 8 条其他请求中的多余追问 | 0 | 0 | 0 | 0 |
| 读错记录 | 0 | 0 | 0 | 0 |

例如用户暂未给出姓名时，模型问「请告诉我要查谁的用品」。回复「郭回霜当前版」后，入口从记录 11 读回「软尺」。定向头也把一句已经明确说了郭回双的地点请求误判为缺姓名。新头跳过了 128 条公开普通请求。这组样本只支持初步评估，缺项题的补答均指定当前版。

首轮给定答案的对照有一条补答「用品当前版」进入普通聊天。旧解析先删除命令前缀，把「用品」开头的「用」误删。现优先解析完整的字段或姓名，失败后再解析带命令前缀的形式。新增回归覆盖裸字段、命令前缀、给出两个字段以及取消后迟到的补答。复跑时新头权重摘要完全一致，给定答案时从 7/8 增至 8/8，失败题现在读取记录 15 并返回「蓝绳」。两轮各约 81 秒，64 项版次测试通过，源模型和档案摘要保持一致。循环版 Windows 启动入口通过原有 24 项回放和重启后的 13 项回放，档案摘要一致。

下一轮应检验已知坐标是否能纠正不必要的追问。把「没有给姓名」「已经明确给姓名」「提到两人但只查其一」组成匹配题，并加入上一轮已确定地址和用户临时改口的中文对话。把定向追问与完整地址提议结合后，在独立题上同时报告缺项恢复和对完整请求的额外追问。 [中文歧义对照研究](https://arxiv.org/abs/2507.23121) · [分解请求歧义与动作置信](https://arxiv.org/abs/2606.19559)

## 复现

逐题回放和权重在 `runs/spora-spoken-coordinate-clarification-v2/`。较早解析的对照在 `runs/spora-spoken-coordinate-clarification-v1/`。

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_spoken_coordinate_clarification runs/spora-read-confirmation-action-v1 runs/spora-crosswoz-field-transfer-v1 runs/spora-dense-joint-address-v1 runs/research/xing-usp-883c7dc runs/spora-spoken-coordinate-clarification-new --recurrent @weights
```
