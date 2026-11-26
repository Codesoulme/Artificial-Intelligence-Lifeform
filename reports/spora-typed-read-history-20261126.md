# 确定地址及版次后的双项读出

上轮有一条双项查阅在重启后漏报书名。另有一次已经改查单项地点，用户说「首版」时，普通生成复述了当前版值。两类失败出现在不同阶段。结构化生成研究指出，如何组织输出以及如何限制生成会分别影响行为。此次从同一份已校验档案重建张量，把地址和版次选择、生成历史以及结束处理分开观察。 [上轮回放](spora-paired-lookup-clarification-20261126.md) · [结构化解码研究](https://proceedings.mlr.press/v306/reddy26a.html) · [约束与动作选择对照](https://arxiv.org/abs/2608.13959)

使用原循环骨干、词表和读出权重。三条完整请求和失败请求都由入口的解析逻辑先解析出两个记录编号，核对编号及从张量重建的值。四条请求保持原文，分别重放当时裸「首版」的普通答复历史和空历史。每种历史里，对比屏蔽过早结束符和在预测过早结束时复制尚未输出的字段。记录正式复制动作、值的词元及结束状态。

| 条件 | 完整复制两项 | 原失败题的结果 | 原失败题生成时间 |
| --- | ---: | --- | ---: |
| 有旧聊天，屏蔽结束符 | 3/4 | 只复制地点，余下预算多为标点 | 3.87 秒 |
| 有旧聊天，遇过早结束补全 | 4/4 | 两项均复制 | 0.89 秒 |
| 空历史，屏蔽结束符 | 4/4 | 两项均复制 | 0.82 秒 |
| 空历史，遇过早结束补全 | 4/4 | 两项均复制 | 0.61 秒 |

上一轮的失败题确定读记录 3 和 2，从张量重建「南风巷」和「远山」。此前的聊天历史里仅有用户的「首版」和模型要求补信息的回答，未出现相关值。旧解码复现到 24 词元上限，模型用标点占掉了第二项复制所需空间。空历史和结束补全各自都能修复这道题。上表为一次本机测量，含每次构造输入和生成，整轮约 151 秒，含模型加载。

另一个裸「首版」对照在普通生成时带入上轮「见面地点：云桥」会复述「云桥」，用空历史则要求更多信息。现在如果上一轮实际只读了一项，用这项已经确认的地址查用户要求的版次。重启、取消或有多个可能地址时，先问具体哪一项。已确定地址和版次的带字段名称查阅从张量生成读回，去掉生成提示中的旧聊天，并采用结束补全。自然写作仍使用单独的生成路径。状态跟踪工作也将载入当前约束和按约束行动分别评估。 [StateWise](https://arxiv.org/abs/2610.33766) · [SAGE](https://arxiv.org/abs/2609.00434)

Windows 循环版从空档案登记十条记录。条件齐全 3/3、缺版次 5/5、取消与改查 4/4、重启 2/2。连同登记和重置，50 个入口检查全部通过。改查单项后答「首版」实际读取记录 3，显示「南风巷」。取消或重启后同一句只请求坐标，未选记录。重启后的完整请求实际读取 3 和 2，显示「南风巷」和「远山」。档案在重启前后摘要一致，SQLite 完整性检查通过。整套回放约 305 秒，含两次模型加载。115 项相关单元测试通过。逐轮结果位于 `runs/spora-paired-lookup-clarification-typed-read-v1/result.json`。

四种生成对照和入口回放使用有限的已登记字段。下一轮可用同一句「首版」配上不同的当前地址、撤回状态和双项范围，训练请求规划头区分可执行与需补问。单独测选址、选版和实际读回，避免把已选张量的覆盖能力计入自然寻址准确率。 [已有模型内会话状态对照](spora-latent-working-state-20261124.md)

## 复现

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_typed_read_history runs/spora-paired-lookup-clarification-candidate-v1 runs/spora-typed-read-history-new --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_paired_lookup_clarification runs/spora-paired-lookup-clarification-typed-read-new --recurrent @weights --require-all
```

对照脚本会再次运行四种读出条件，并调用当前磁盘上的入口。原始逐题结果位于 `runs/spora-typed-read-history-v1/result.json`。
