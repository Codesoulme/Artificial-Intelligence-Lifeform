# 三层循环候选的中文答复概率对照

此前比较四种双层布局，发现第 19、22 层的中文长对话表现较好。本轮使用已对齐的第 21、24 层权重，分别追加到这两个已迁移层，先衡量第三处结构替换对真实中文答复的影响。FlashMorph2 在整网离散结构上评价候选，并指出恢复训练会改变布局排序。多语言混合注意力研究也观察到循环层和精确注意力层的排列改变语言表示。 [既有布局实验](spora-hybrid-layer-placement-20261125.md) · [FlashMorph2](https://arxiv.org/abs/2611.24809) · [Multilinguality in Hybrid Attention LLMs](https://arxiv.org/abs/2609.35378)

## 对照

从 CrossWOZ 固定版本的官方测试档案中按会话编号摘要确定采样顺序。每份对话取前三轮，前五条消息组成预测上下文，最后一条真实系统答复作为监督目标。保留 80–384 词元上下文及 10–96 词元的回答。先选 12 段验证流程，再扩展到 48 段，共 1,115 个实际答复词元。四个布局使用同一个父检查点，比较无替换、现有双层、两个三层候选。各布局直接执行整网前向及其实际状态更新，读取给定前文和正确答案前缀时的下一词元对数概率和首选词元。此轮没有更新模型权重。CrossWOZ 测试分区曾用于此前字段提取评估，本轮未使用该分区训练迁移层，不能据此断言整条底座训练来源完全无重合。

| 布局 | 前 12 段平均负对数概率 | 48 段平均负对数概率 | 48 段首选词元正确率 |
| --- | ---: | ---: | ---: |
| 父模型 | 5.1566 | 4.8842 | 30.22% |
| 19、22 | 5.2004 | 4.9642 | 28.34% |
| 19、21、22 | 5.1728 | 4.9736 | 27.80% |
| 19、22、24 | 5.3022 | 4.9760 | 27.53% |

扩大样本后，两种第三层候选的平均损失均略高于当前双层版，第 21 层候选在首批表现出的损失优势没有延续。新增 36 段中，第 24 层平均损失为 4.8989，双层为 4.9085，首选词元正确率分别为 28.71% 和 29.60%。因此两种度量并未得出完全一致的排序。48 段实验约 71 秒。原 12 段四组逐题结果完全复现，个体和源权重摘要一致。10 项循环注意力单元测试通过。

这是给定正确答案前缀的语言保真诊断，尚未测量第三层候选的实际自由生成或具名持久记忆读取。两种第三层候选的损失差距很小，暂时保留双层为默认布局。下一轮对两个布局进行同预算的整网预测恢复，保留词元及完整答复对照，再测具名持久读取和中文多轮生成。 [记忆路径协调研究](https://arxiv.org/abs/2610.06750)

## 复现

逐题结果位于 `runs/spora-hybrid-language-likelihood-v2/result.json`。

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_hybrid_language_likelihood runs/spora-composition-late-gates-20-23-24-v1/candidate.pt runs/research/crosswoz-df82c9f/test.json.zip runs/spora-hybrid-language-likelihood-new/result.json runs/spora-gelu24-recurrent-layer19-v2 runs/spora-gelu24-recurrent-layer21-morph-v1 runs/spora-gelu24-recurrent-layer22-v1 runs/spora-gelu24-recurrent-layer24-morph-v1 --count 48
```
