# 原版 DeepSeek 的中文原生推理对照

上一轮的 215 词元上限让原版 DeepSeek 的 20 次推理中有 16 次被强制结束，孢子是 18 次。本轮用原版 DeepSeek-R1-Distill-Qwen-1.5B 本机权重单独对照，检验允许模型自然结束推理，是否足以改善日常中文成品。另起八道题，含五道单轮和三道附一组现成前文的题。题目和通过标准定义在脚本中。 [上轮实验](spora-open-reasoning-20261125.md) · [DeepSeek 官方解码建议](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B#usage-recommendations)

两组采样均使用温度 0.6、top-p 0.95、逐题相同的种子和 128 词元正文上限。直接作答模式在输入末尾结束推理段。原生模式最多生成 768 个推理词元，由模型生成结束标记后再开始计正文预算。检测到不合规范的重复边界时不输出这次正文。直接贪心作为参照。只保留可见答案、词元数量、终止标记和耗时。

| 模式 | 有正文 | 正文正常结束 | 严格人工通过 | 耗时中位数 |
| --- | ---: | ---: | ---: | ---: |
| 直接贪心 | 8/8 | 8/8 | 0/8 | 2.82 秒 |
| 直接采样 | 8/8 | 7/8 | 0/8 | 4.33 秒 |
| 原生推理采样 | 8/8 | 7/8 | 1/8 | 42.42 秒 |

原生推理在 8/8 题上自行结束，耗用 255–577 词元，中位数为 334.5。开始正文前等待的中位数为 37.78 秒，峰值 CUDA 张量约 3.36 GiB。它写出一条可直接发送的改期提醒，准确保留邮局和雨伞，替换旧时间。直接采样的同题保留了事实，但加了引导语和额外措辞。原生推理仍复述了明确要求省略的手提灯，把前台误译成门，还把有风加速蒸发解释成灰尘的作用。问时间和地点时，它在实际问题外增加了引导语和说明。严格评分按整条答复判断，样本很小，采样每题只有一个种子。此轮只测原版，不能据此估算孢子结构中的推理收益。

近期的动态早停研究使用候选答案的置信度判断何时结束，PUMA 还加入对推理内容是否重复的判断。其验证任务主要衡量已有答案能力下的节省。本轮允许原版自然结束推理之后，开放题仍有七题未通过。当前证据不足以支持将早停或更长推理预算作为日常对话的主要改造。保留这轮作生成对照，结构实验继续定位跨层的查询与读出。 [Dynamic Early Exit in Reasoning Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8df90a1440ce782d1f5607b7a38f2531-Abstract-Conference.html) · [PUMA](https://arxiv.org/abs/2605.17672) · [循环注意力布局](spora-hybrid-layer-placement-20261125.md)

逐题输出与源文件摘要见 `runs/spora-natural-reasoning-v1/result.json`。在同一套本机权重和环境中可复现。

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_natural_reasoning runs/spora-natural-reasoning-new/result.json
```
