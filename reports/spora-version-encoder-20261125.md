# 用轻量中文编码器比较版本意图识别

四选一提示在 8 组新措辞对照中一组也未完整选对。本轮暂停提示词迭代，换用冻结的中文语义编码器。TDM-Gauntlet 记录了英语任务上标签定义不足时的零样本分类差距。TIDE 将隐含时间与版本不适用分别评分。两篇研究提供了实验思路，效果以本机中文数据为准。 [四选一结果](spora-version-menu-20261125.md) · [TDM-Gauntlet](https://arxiv.org/abs/2611.20191) · [TIDE](https://arxiv.org/abs/2608.08512)

## 设置

在 CPU 上比较 BAAI `bge-small-zh-v1.5` 和一至二字词频。两边使用同一批 200 条训练、52 条校准问句和同一种两头分句分类器，分别判断分句的操作含义与指向的版次，各更新 180 步。分类器只看问句。现有时间轴处理日期和明确版次。本轮只评估读取决策，使用独立实验文件，无需载入 DeepSeek 或个体检查点。编码器权重约 96 MB，使用 MIT 许可的固定版本。 [模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/7999e1d3359715c523056ef9478215996d62a620)

复用四选一的 172 条评测题。另造 60 条措辞留出题，固定人物及事项地址，专测句意到时间坐标的映射。同一对题互换请求与排除所修饰的版次，也分别测试请求句在前和在后。三种版次两两组合。60 题共用一组新表达和四个事项，实际只有 24 组成对比较，且部分重复语言结构。阈值仅由校准集决定，先减少错误读取，再增加正确读取。

## 结果

| 测试集 | 原方案正确／误读 | 字符正确／误读 | BGE 正确／误读 | BGE 加阈值正确／误读 |
| --- | ---: | ---: | ---: | ---: |
| 四选一新题，32 条 | 12／4 | 24／4 | 28／0 | 16／0 |
| 操作词位置，64 条 | 15／15 | 16／32 | 40／8 | 16／0 |
| 已有回归，76 条 | 68／0 | 64／8 | 68／4 | 60／0 |
| 额外留出，60 条 | 12／0 | 0／20 | 24／20 | 18／0 |

正确包含应暂缓时未读取。误读包含读错版本和本应暂缓却读取。BGE 在四选一新题中读取正例 12/16，成对选对 4/8。额外留出题读取正例 16/48，成对选对 0/24，12 条负例误读 4 条。阈值取 0.95 后，只保留额外留出题的 6/48 条正确读取，还挡掉了上一批全部 12 条正确读取。

抽查显示，「今天办事依照的那条」被判成首版，「留待下回再核」有时被判成查阅。当前试验保留为对照，运行入口采用原选择策略。下一阶段停止围绕本组措辞调阈值，先准备按说法家族隔离的人工复核中文样本。每条标出所请求的版次、被排除的版次和说话人的操作，最小对照只替换其中一项。同时加同义改写和只有排除、没有请求的负例。ChronoQG 可参考的是怎样用固定时间关系生成可验证的成对问题。 [ChronoQG](https://arxiv.org/abs/2607.14770)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$menu = 'runs/spora-version-menu-v1/result.json'
$timeline = 'runs/spora-version-timeline-v2/versions.pt'
$encoder = 'D:/SPORA-runtime/models/bge-small-zh-v1.5-7999e1d3'
D:/SPORA-runtime/venv/Scripts/python.exe -u -m scripts.audit_spora_version_encoder $menu $timeline $encoder runs/spora-version-encoder-v1.json
```

初次 CPU 执行约 2.6 秒，另行运行逐项复现了 232 条决策、置信度、汇总和输入摘要。测试输出保存在 `runs/spora-version-encoder-v1.json`，复跑保存在 `runs/spora-version-encoder-v1-reload.json`。
