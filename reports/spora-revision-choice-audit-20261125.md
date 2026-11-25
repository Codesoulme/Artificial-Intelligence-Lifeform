# 中文更正对照：能否辨认仍然生效的要求

上一轮发现，接入孢子结构的千问在六条留出任务上分轮答对三条，给出整理好的现行要求后答对五条。本轮冻结模型，直接测它是否能辨认哪项要求仍然生效。原 DeepSeek 孢子和采用千问指令版权重的结构对照均参测。 [前一轮结果](spora-dialogue-backbone-pivot-20261125.md)

## 方法

新写八组日常中文对话。每组共享前两轮和两段候选内容，只改变最后一句，正确候选随之交换。题目包括撤回、反例引用、备选采纳、条件口令和恢复旧版。判断只针对两个候选争议的那项内容。各题分别给出三轮、只给末轮和省略用户发言。省略发言后，每组对话的两种答案各出现一次，所以固定选择只能答对半数。

第一个测量把内容摆成 A 和 B，比较两个标签在回复起点的分数。对同一题交换位置再测，把差值统一到同一内容方向后取平均。PriDe 研究表明模型常有选项标签偏好，固定顺序的分数需要配合反向顺序解读。每组另统计最后一句反转时是否两题都选对。两个底座各用原生聊天格式，DeepSeek 采用现有聊天入口的直接回答前缀。 [PriDe](https://arxiv.org/abs/2309.03882) · [StateMemBench](https://arxiv.org/abs/2608.19652)

第二个测量将候选文本分别作为回复前缀，比较内容逐词元的平均对数概率，再分别减去省略用户发言时的分数。它衡量用户发言给每个候选增加了多少支持。处理方法借鉴无上下文先验校正与对比解码，应用范围是给定候选内容的排序。 [表述竞争与概率校正](https://arxiv.org/abs/2104.08315) · [上下文对比解码](https://arxiv.org/abs/2305.14739)

## 标签选择结果

| 模型 | 三轮固定顺序 | 三轮对调后 | 双顺序合并 | 两题均通过 | 两个顺序选同一内容 |
|---|---:|---:|---:|---:|---:|
| 当前 DeepSeek 孢子 | 8/16 | 8/16 | 8/16 | 1/8 | 0/16 |
| 千问孢子结构对照 | 8/16 | 9/16 | 8/16 | 0/8 | 9/16 |

DeepSeek 的三轮题在两个顺序下都把 A 的分数排在 B 前面。双顺序合并后只有借还地点一组两题都选对。千问在九题上交换选项后仍选同一内容，但合并分数后每组恰好答对一题。仅给末轮时，双顺序合并分别答对 5/16 和 9/16。固定答案和不看发言的对照均为 8/16。

## 候选文本评分结果

| 模型 | 三轮文本评分 | 仅末轮 | 不给发言 | 三轮分数减先验 | 两题均通过 |
|---|---:|---:|---:|---:|---:|
| 当前 DeepSeek 孢子 | 9/16 | 6/16 | 8/16 | 9/16 | 1/8 |
| 千问孢子结构对照 | 9/16 | 7/16 | 8/16 | 9/16 | 1/8 |

去掉字母后，两版仍在多数成对任务上坚持同一项选择。例如用户把备选标题从仅供参考改为正式采用，两版按内容打分仍选原定标题。DeepSeek 的上下文校正做对了柜台方向的两种情况，千问做对了恢复原词和采用新词的两种情况。两种评分只能在给定候选之间选择，评测只含用户发言，尚未覆盖从自由对话自行定位、抽取和修正条件。

结果支持在模型内部直接学习「哪条来源仍然有效」。下一轮可在相同历史与候选之间加入只改末轮的成对监督，同时区分采纳、保留、撤回和引用。先验证两种末轮能否分别选对，再检验自然生成中的旧要求泄漏。FollowBench 的中文内容和格式数据提供现成任务素材，其原始任务逐级增加约束，撤回和恢复的配对需要另行编写及复核。 [FollowBench 中文数据](https://github.com/YJiangcm/FollowBench/tree/master/data_zh) · [AFVE-Bench](https://arxiv.org/abs/2611.10541)

四次独立执行共约 351 秒，DeepSeek 的 CUDA 张量峰值约 3.56 GiB，千问对照约 2.95 GiB。三个现有对话状态测试通过。逐题分数、预测和模型标识存于 `runs/spora-revision-choices-v1/`，本轮没有训练新权重。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$out = 'runs/spora-revision-choices-v1'
$checkpoint = 'runs/named-memory-layer9-packed-reply-v1/packed.pt'
$donor = 'D:\SPORA-runtime\models\Qwen2.5-1.5B-Instruct-989aa798'
foreach ($score in 'labels','content') {
    .\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_choices $out --model deepseek --score $score
    if ($LASTEXITCODE) { break }
    .\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_choices $out --model qwen --score $score --checkpoint $checkpoint --donor $donor
    if ($LASTEXITCODE) { break }
}
```
