# 中文聊天底座与孢子改造版同场对照

## 条件

本轮用两组开发材料量化同一个问题：原版 DeepSeek 在本机直接回答设置下，能否遵循多轮中文约束，以及现有孢子改造额外改变了多少。所有模型逐一加载到同一张 GTX 1060 6 GiB，使用 FP16、贪心生成和每轮最多 64 个词元。各自使用原生聊天模板，DeepSeek 在模板末尾关闭思考段，千问指令版直接作答。各模型使用各自的正常结束标记，重复惩罚统一为 1。计时从已完成分词的输入开始，并在计时前完成一次预热。

- **固定历史 24 题。** 每个模型接收完全相同的两轮人工对话和第三轮问题。三类各八题，分别检查更正、承接较早约束和承接最新约束。仅在忽略首尾常用标点后与目标短答完全一致时计为通过。
- **自行续聊 24 轮。** 复用 `conversation-development-v2` 的八个三轮场景。模型真实回复进入自己的下一轮，检查如何处理早餐限制、时间更正、人物关系与未知事实。自然结束记录生成是否停止，语义表现需看逐题输出。
- **孢子个体。** 使用日期记忆开发入口的检查点与个体，在普通对话生成时关闭持久记忆读出。实验核对模型、数据文件和加载后神经状态的摘要。另有带日期记忆读出的同检查点验证，见 [日期记忆与普通对话](spora-mixed-calendar-chat-20261122.md)。

## 同场结果

| 模型 | 更正 | 较早约束 | 最新约束 | 严格短答合计 | 自行续聊自然结束 | 自行续聊中位生成时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原版 DeepSeek-R1-Distill-Qwen-1.5B | 0/8 | 0/8 | 0/8 | 0/24 | 17/24 | 1.75 秒 |
| 日期记忆孢子检查点 | 0/8 | 0/8 | 0/8 | 0/24 | 18/24 | 2.10 秒 |
| 千问 Qwen2.5-0.5B-Instruct | 1/8 | 2/8 | 2/8 | 5/24 | 21/24 | 0.86 秒 |
| 千问 Qwen2.5-1.5B-Instruct | 4/8 | 6/8 | 2/8 | 12/24 | 20/24 | 1.32 秒 |

固定历史中，原版 DeepSeek 和孢子有 20/24 条可见回复逐字一致。原版在一道题里建议免费社区展览，孢子考虑两个选项而漏掉免费约束。原版在一道题里建议下雨时去室内大厅，孢子改为拒答。两个原版答案均附带解释，因此严格得分未体现这两处语义差异。两份模型在自行续聊中都给出了含蛋的早餐方案，都没维护周四改约，也都建议在下雨时调整骑行。原版的问题能在关闭孢子记忆的对照中复现，当前改造仍有逐题退步。

千问 0.5B 能正确完成上午和下午地点的三轮对调，也能生成无蛋早餐。它把杂志借出者答反，声称自己又数了一遍用户抽屉里的纸，也没正确更新改约。

千问 1.5B 更新了周四晚上八点的改约并保留影院门口，能算出七张纸拿走两张剩五张，也明确拒绝猜测人物没喝咖啡的原因。它仍将杂志借出者答反，且有三道固定历史题恰好输出旧答案。四个模型的峰值 CUDA 分配分别为 3.34、4.94、0.95、2.91 GiB，顺序同表。

在本机短答条件下，DeepSeek 的日常会话约束问题已经出现在原模型中。孢子结构还有语义回退，范围可用固定历史逐题查出。千问 1.5B 可作为指令能力对照，正式结构目标仍沿用 DeepSeek 孢子模型。

用提交 `eb81446` 的隔离源码、同一检查点及本轮脚本重跑孢子。固定历史和自行续聊共 48 条记录复现可见回复、词元数、结束标记和短答判定。检查点及个体文件摘要与运行前一致。另有两项日期记忆对话入口的单元测试通过。

## 孢子对话诊断

用两条给定续写计算平均词元对数概率时，孢子在完整历史下有 18/24 题把正确短答排在干扰项前。对应的自由生成严格通过 0/24。仅保留最近一轮时严格通过 1/24，正确选项排列领先 14/24。在末轮问题前加一条提醒时严格通过 1/24，正确选项排列领先 16/24。这三组结果来自同一孢子检查点。

有些生成包含正确内容，例如将「南门」写成「现在要走的门是南门」。有些复述指令或给出固定拒答。成对排序测量指定答案之间的相对偏好，开放生成仍需要从完整词表内选出正确起点，保持答案格式并结束。提高已知正确答案对旧答案的相对分数，未必会处理这些自由生成失败。

## 适用范围

这次比较的是各底座在本机短回复模式下的表现。24 道短答对格式敏感，自行续聊场景属于开发材料。各发布模型的训练配方、模型参数和对话模板都有差异，结果可比较本机使用表现，无法估计其中任一因素的独立贡献。较早的推理模式实验已单独比较 DeepSeek 的原生思考模板，其生成设置和题目不同。[本机生成预算对照](spora-reasoning-budget-20261122.md)

Qwen2.5 的技术报告描述了针对指令遵循的后训练，DeepSeek-R1 报告的蒸馏目标重点覆盖推理行为。当前孢子结果说明，迁移时需要同时测量预训练语言函数的保持程度和底座本身的日常对话水平。[Qwen2.5 报告](https://arxiv.org/abs/2412.15115) · [DeepSeek-R1 报告](https://arxiv.org/abs/2501.12948)

后续实验沿用 DeepSeek 孢子检查点，针对学生自己生成的对话历史收集约束遗忘和更正失败。先在独立训练场景中人工复核千问给出的修正回复，再考虑将其作为离线监督。对比同训练预算的原方案，并以逐题日期记忆回放和独立多轮对话共同验收。这借鉴了 GKD 在学生生成历史上提供教师反馈的思路。本轮交付的是评测结果。[On-Policy Distillation](https://arxiv.org/abs/2306.13649)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$state = @('--checkpoint', 'runs/spora-native-query-key-v1/bootstrap.pt', '--database', 'runs/spora-calendar-sleep-v1/individual.sqlite')
$fixed = 'runs/spora-dialogue-constraints-v1/result.json'
.\.venv\Scripts\python.exe -u scripts/audit_spora_dialogue_constraints.py @state --output $fixed
.\.venv\Scripts\python.exe -u scripts/audit_spora_chat_backbones.py --fixed $fixed --output runs/spora-chat-backbones-v1/deepseek.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_chat_backbones.py @state --fixed $fixed --output runs/spora-chat-backbones-v1/spora.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_chat_backbones.py --qwen D:/SPORA-runtime/models/Qwen2.5-0.5B-Instruct --fixed $fixed --output runs/spora-chat-backbones-v1/qwen-0.5b.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_chat_backbones.py --qwen D:/SPORA-runtime/models/Qwen2.5-1.5B-Instruct-989aa798 --fixed $fixed --output runs/spora-chat-backbones-v1/qwen-1.5b.json
```

千问 1.5B 使用官方模型仓库的固定版本 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，权重 SHA-256 为 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。[官方模型页](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)
