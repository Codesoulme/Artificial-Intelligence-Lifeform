# P235：长对话切分与来源调查

P231 的 68 段候选因整段超过 512 词元而排除。`scripts/prepare_p235_long_dialogues.py` 固定原训练侧、索引、族划分和词表摘要，对这部分逐个助手回答构造候选：保留完整当前用户与助手轮次，优先保留最近的完整历史；超长时仅移除最早整轮，不裁切回答。历史助手回答只作为上下文，监督掩码只覆盖当前回答和结束符。单轮本身超过窗口则跳过。源行、族、对话摘要、轮次及上下文起点均保留，本次未使用最终测试或训练。

68 段共 160 个助手轮次；21 个单轮过长而跳过，保留 139 轮、32,313 个监督目标。其中 111 轮没有保留较早历史。同一对话的轮次相互依赖，材料及其祖先接触、事实、隐私和权利均未完成核验；这些数字不是独立或合格的新训练量。候选文件与清单位于 `runs/p235-long-dialogue-candidates-20261109/`。

`scripts/sample_p235_long_dialogues.py` 排除 P231 抽查的族，按固定散列每族选一轮、共 20 轮，附保留前的原有上下文，样本位于 `runs/p235-long-dialogue-review-20261109/`。两位模型各审 10 条：2 clean、5 minor、13 major；8 条判保留上下文足够、11 条不足、1 条不确定。所有 20 条的目标解码与原回答加结束符逐字一致；已核对评阅序号、族和对话摘要。样本按族而非按目标抽取、两位审阅者各看不同条目，结果不是交叉复核、人工判定或整个候选池的质量估计。主要问题为摘要和提取缺少原文，以及事实错误、无根据的现实经历或行动承诺。模型评语本身未做外部事实核验。

结论：不把切分结果直接投入训练；保留完整原上下文并建立上下文依赖审查，才能判断这些片段是否适合下一阶段。对事实、身份和约束的筛选仍须在新样本验证，不能将开发样本的词句直接当作通过规则。`tests/test_prepare_p235_long_dialogues.py` 覆盖完整答案、超长跳过、历史剪裁和旧答案掩码；相关测试共 13 项通过。

## 来源线索

公开资料调查找到 [NaturalConv](https://huggingface.co/datasets/xywang1/NaturalConv/blob/adb38574e8014ada1d8404247167582e76b22d05/README.md)、[DuConv](https://huggingface.co/datasets/PaddlePaddle/duconv/blob/2080deae0c89256bb023ad321b453dec5971b61a/README.md) 和 [LCCC-base](https://huggingface.co/datasets/thu-coai/lccc/blob/9022ba27075f75c2f59d57d7fa5f42e8d1151aec/README.md)。NaturalConv 为主题引导下的人类长对话，但[发布条款](https://huggingface.co/datasets/xywang1/NaturalConv/blob/adb38574e8014ada1d8404247167582e76b22d05/LICENSE)限个人非商业用途并要求由腾讯来源下载；DuConv 是围绕知识与目标的众包对话，卡片标 Apache-2.0，其引用的压缩包为可变外链；LCCC-base 来自社交对话，规模大但个人信息及来源权利需要单独核验，卡片的 MIT 文本针对软件。它们都不能因公开或卡片标签被直接视为已核权、无隐私风险或适合作为助手示范。本轮仅查看公开说明，没有下载上述语料或将其用于训练；与项目现有及祖先材料的实际重合尚未知。
