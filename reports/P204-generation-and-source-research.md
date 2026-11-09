# P204 生成方式与数据来源核查

## 研究依据

- [TinyStories](https://arxiv.org/abs/2305.07759) 报告以简单英文合成故事训练小模型可产生连贯叙事；论文的数据、英文任务及训练设置与本项目的中文翻译小样本不同，不能直接推算本机所需步数或预期能力。
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) 研究发现同一模型的生成方式会影响重复及多样性，并提出 nucleus sampling。这不表明采样能修复尚未学到的中文语义或对话能力。
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) 强调训练计算和数据量的关系；其实验模型及数据量远大于本机，不能把论文的比例直接作为本项目的最佳训练配方。

## 固定模型对照

同一个 P203 检查点，使用此前四条开发提示，各生成最多 48 个单位，只屏蔽 BOS/PAD。贪心每提示一次；nucleus 使用 `p=0.9`、温度 1、每提示两个固定种子，不进行 UTF-8 修复或重复惩罚。

| 方式 | 条数 | 合法 UTF-8 | 正常结束 | 相邻相同单位合计 |
| --- | ---: | ---: | ---: | ---: |
| 贪心 | 4 | 4 | 0 | 31 |
| nucleus | 8 | 7 | 0 | 7 |

采样减少了本次的相邻重复，但可解码的续写仍普遍语序混乱、情节断裂，问候也没有得到有效回答；一条含非法字节，12 条均耗尽预算。相邻重复数只是表面指标，样本小、种子有限，提示已反复用于开发，不能据此评价总体效果。所有原始单位和可解码文本保存在 `runs/p204-generation-mode-audit-20261109.json`。输入文件和冻结个体状态在对照前后摘要一致；未读取测试正文或修改权重。

## 新候选来源

通过 Hugging Face 元数据和固定版本数据卡核查：

- [`richarddzh/chinese-small-lm-corpus`](https://huggingface.co/datasets/richarddzh/chinese-small-lm-corpus/tree/c725144fe086c28d8d8a5994fda86e9e23046fde)，修订 `c725144fe086c28d8d8a5994fda86e9e23046fde`，卡片声称 438 万条，含 TinyStories 中文翻译、2023 中文维基百科和 Zhihu-KOL，字段包括 `text` 与 `source`。卡片明确不提供统一再授权，Zhihu 上游未声明许可；维基内容须按对应条款处理，故事与既有 P28 可能同源。未下载正文或用于训练。
- [`RobinChen2001/TinyStories-Zh-2M`](https://huggingface.co/datasets/RobinChen2001/TinyStories-Zh-2M/tree/7601b862e5b2fbf60e83d7b27b8bd8017f105e70)，修订 `7601b862e5b2fbf60e83d7b27b8bd8017f105e70`，卡片标注 MIT，并说明它整理自 `52AI/TinyStoriesZh`，英文来源为 TinyStories。不能把卡片标记当作全部上游与翻译权利的证明，也不能把同源故事算成独立新增量。
- [`n-pn/chinese-novel-corpus`](https://huggingface.co/datasets/n-pn/chinese-novel-corpus/tree/a341c4b5403b48002ab0e09b74fcdc0a6cc24a93)，修订 `a341c4b5403b48002ab0e09b74fcdc0a6cc24a93`，README 仅标记 Apache-2.0，未说明各作品和版本的来源权利。不能凭汇总许可认定小说正文均可使用，未下载或训练。

数据集预览接口在本次请求中超时，因此这些是数据卡和元数据结论，不是正文质量评估。以上候选保持未批准。下一步应优先寻找权利与来源可追溯、文体更自然且有明确文档族的材料，并通过实测区分数据、训练预算和模型结构的影响。
