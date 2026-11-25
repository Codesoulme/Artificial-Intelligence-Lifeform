# 在本机聊天中确认口语查阅的范围

上轮独立版次头提升了少量口语查阅，也出现了高分错读。这轮把已训练的查阅头接入循环版 Windows 聊天入口，让它在口语请求中提出人物和事项，再让用户确认地址并给出版次。ToolTurns 用先澄清再操作的完整交互衡量成功。TIDE-Bench 区分层层补充信息和撤回旧目标。StateGraph 也分别处理原结论与它依赖的前提。 [上轮版次实验](spora-spoken-temporal-head-20261126.md) · [ToolTurns](https://arxiv.org/abs/2611.14925) · [TIDE-Bench](https://aclanthology.org/2026.emnlp-main.2454/) · [StateGraph](https://arxiv.org/abs/2611.02912)

## 实现

`--spoken-lookup` 为循环版启用独立试验。它验证公开负例头与当前父检查点、两个循环注意力替换权重的摘要。只有既有解析器未取得查阅目标时，才用头的查阅动作、人物和事项分数提出范围。普通翻译、编辑、明确取消及转题继续由现有机制处理。阶段一只返回确认问题，选中记录列表为空。用户可以用「是，上一版」同时确认地址和版次，也可以分两轮补齐。只说「上一版」会再次提示确认。明确改口能提供新的姓名、事项和版次。确认后复用原来的版次解析和张量读取。

待确认状态仅保存在当前会话。否认、取消、转题、清空及退出都会释放它。新提议先清掉之前的查阅范围和对话文本，后续回复的读取范围由当次确认决定。

测试从上轮已登记的四条三版时间线备份出独立档案，运行常规 Windows 启动脚本。核对每轮类型、提议的人物与事项、实际读取编号及结果包含的登记值。跨重启核对源档案摘要及副本记录内容。

## 实测

第一次运行的 31 项检查和重启后的 6 项检查均通过。五次明确读取都选对了档案编号。比如模型先提议查罗映禾的见面地点，用户说「不是罗映禾的见面地点，是罗映荷的饮品首版」，入口随后读取罗映荷的首版饮品「山楂茶」。单报「上一版」的第一次回应仍在等待确认。取消、拒绝、转题、翻译、清空和重启后的迟到确认均没有读取已撤回的范围。这两次分别约 51 秒和 48 秒，含加载。

五条新口语问法均落入普通对话。按是否适合提问重新校准时，153 条既有校准请求中有 48 条要求查阅。0.90 阈值给出 5 个正确地址，零错误或多余提议。降到 0.80 后有 7 个正确地址、2 个错地址和 1 个多余提议。按正确数减去两倍错误数计分，最后仍选 0.90。因此这轮获得了可交互验证的确认流程，触发覆盖仍然很窄。校准代码和逐阈值结果位于 `runs/spora-spoken-proposal-v1/result.json`。

使用校准产物复跑，31 项及重启后 6 项再次全部通过，五条新问法的结果相同。第二次约 45 秒和 44 秒，含加载。不开启口语试验时，循环版的原确认流程也通过首次 24 项及重启后 13 项。61 项版次单元测试通过。逐轮记录位于 `runs/spora-spoken-lookup-dialogue-v1/result.json`、`runs/spora-spoken-lookup-dialogue-v2/result.json` 和 `runs/spora-lookup-clarification-spoken-default-v1/result.json`。

单独查看五条新问法的输出分数时，人物和事项的最高分答案均正确。动作头把三条分为继续聊天，一条分为澄清，一条分为读取且分数为 0.72。在「罗映荷先不用问，罗映禾约我在哪儿见面？」中，它在人物上选了罗映禾，却以 0.996 的分数判定整句无需查阅。下一轮优先训练动作头识别否定与确认各自作用的句段，例如比较「第一人不用查，第二人仍需要查」和「两个都不需要查」。独立验证保留普通中文请求、相似姓名以及同句翻译。Xing USP 支持分开检查动作与人物和字段绑定，读取许可研究强调区分漏掉证据和误打开记录。 [Xing USP](https://arxiv.org/abs/2610.34343) · [读取许可验证](https://arxiv.org/abs/2610.07309)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
$proposal = 'runs/spora-spoken-proposal-new'
.\.venv\Scripts\python.exe -u -m scripts.calibrate_spora_spoken_lookup runs/spora-spoken-query-data-v2 $proposal --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_spoken_lookup_dialogue runs/spora-spoken-temporal-head-v1 $proposal runs/spora-spoken-lookup-dialogue-new --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_clarification runs/spora-lookup-clarification-spoken-default-new --recurrent @weights
```
