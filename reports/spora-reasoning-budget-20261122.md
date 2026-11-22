# DeepSeek 原模板与孢子生成预算

在原版 DeepSeek-R1-Distill-Qwen-1.5B 和八层局部注意力孢子检查点上运行同一批 4 个三轮中文场景。每轮保留本模型自己生成的可见答案，隐藏推理文本不返回到对话历史。官方模板以 `<think>\n` 结束。原生条件沿用该起点，采样温度 0.6、top-p 0.95，每轮最多 256 词元。两个直接回答条件显式结束思考段，各 48 词元，分别试贪心和相同采样参数。两种生成路径的上限不同，因此本表比较完整的运行设置。官方榜单使用的生成长度远高于本机试验，四组题每组仅运行一个种子。 [DeepSeek 使用建议](https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations)

| 模型 | 设置 | 严格短答 | 有可见答案 | 自然结束 | 中位每轮耗时 |
|---|---|---:|---:|---:|---:|
| 原版 | 直接贪心 | 0/10 | 12/12 | 9/12 | 1.34 秒 |
| 原版 | 直接采样 | 0/10 | 12/12 | 8/12 | 1.73 秒 |
| 原版 | 原生模板采样 | 1/10 | 6/12 | 6/12 | 16.09 秒 |
| 孢子 | 直接贪心 | 0/10 | 12/12 | 9/12 | 1.46 秒 |
| 孢子 | 直接采样 | 0/10 | 12/12 | 8/12 | 2.51 秒 |
| 孢子 | 原生模板采样 | 0/10 | 5/12 | 4/12 | 17.62 秒 |

开放思考段后，两种模型都给出了木盒棋子的三个正确数量。直接模式只在首轮给出正确数量，后两轮在达到生成上限前都未给出正确的最终数量。原生模板下，这三轮仍未遵守只答数字的格式，孢子的第二轮还撞到词元上限。原版在教室更正后答对五号，下一轮说回原教室却仍答五号。孢子最后一轮答出「三号」，漏了教室名，前两轮没有可见正文。过夜旅行三轮两种模型都耗尽生成预算。本机实测中，打开思考段大幅增加了延迟，短答通过率依然偏低。

此次约 548 秒，峰值 CUDA 分配 4.94 GiB。完整可见回复、结束标志和计时见 `runs/spora-reasoning-budget-v1/result.json`。

## 扩展检查点的位置上限

八层父模型的共享位置混合系数由原静态 RoPE 初始化，检查点中没有保存训练后的共享系数。第 18 层的查询、键和值共用一个可学习绝对位置表，实测其末尾 32 行全零。本轮按这一初始值为表追加零行，从 768 扩至 1536 个位置。保存时核对原检查点完整性。重载时旧行逐元素一致，其他张量和元数据仅有位置上限不同，个体摘要一致。检查点增加约 2.25 MiB。该扩展初始化没有长位置训练。

| 输入长度 | 分段和整段的首选词元一致 | 键值张量 |
|---:|---:|---:|
| 768 | 是 | 16.99 MiB |
| 769 | 是 | 17.01 MiB |
| 896 | 是 | 19.49 MiB |
| 1024 | 是 | 21.99 MiB |
| 1536 | 是 | 31.99 MiB |

与父模型比较，128、512、768 长度的末位置完整词表分数逐元素相同。约 1030 词元的三道临时站名题找回篇首和近末尾的站名，中段题答错。约 1024–1027 词元的 8 道受控记忆题精确答对 7 道，评测给出人物地址与事件版本。由此验证了跨旧上限的计算、续算和有限读出。自然寻址与任意位置的长文能力仍需独立检验。复验约 216 秒，峰值 CUDA 分配 4.94 GiB。

候选为 `runs/spora-native-context-1536-v1/bootstrap.pt`，SHA-256 为 `9f242de8118fa6b80d80d4ac6bf55929d97131d45bc3d2ed05b6b62680ed01c6`。逐项证据见同目录 `result.json`。

## 为最终回答预留词元

用扩位候选重跑原生采样，保持 256 个生成词元和原种子。前 191 个生成词元尚未结束思考时，插入一个结束标记，留 64 个词元继续作答。两次对照的最大输入加生成预算均为 448 个词元，始终在旧位置表范围内。

在 12 轮中强制结束了 7 次。有可见文本从 5/12 增到 12/12，严格短答从 0/10 增到 1/10，自然结束从 4/12 增到 6/12。增加的严格短答是改到五号教室。多条新增可见文本沿着前一段推演继续写，没有形成可用回复。中位每轮耗时从 17.62 秒升到 19.14 秒。试验用时约 304 秒，先保留这条预算策略作为对照。证据见 `runs/spora-reasoning-budget-reserve-v1/result.json`。

扩位的合成回归验证了源文件不变、拒绝已有训练共享系数及非零末行。另有 23 项检查点与局部注意力测试通过，预算处理器的 CPU 检查确认已正常结束的序列不受干预。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -u scripts/audit_spora_reasoning_budget.py --checkpoint runs/spora-anchored-native-eight-v1/bootstrap.pt --budget 256 --output runs/spora-reasoning-budget-v1/result.json
.\.venv\Scripts\python.exe -m scripts.expand_spora_context --checkpoint runs/spora-anchored-native-eight-v1/bootstrap.pt --limit 1536 --output runs/spora-native-context-1536-v1/bootstrap.pt
.\.venv\Scripts\python.exe -u scripts/audit_spora_context_expansion.py --parent runs/spora-anchored-native-eight-v1/bootstrap.pt --checkpoint runs/spora-native-context-1536-v1/bootstrap.pt --output runs/spora-native-context-1536-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_reasoning_budget.py --checkpoint runs/spora-native-context-1536-v1/bootstrap.pt --budget 256 --native-only --spora-only --reserve-answer 64 --output runs/spora-reasoning-budget-reserve-v1/result.json
```
