# 局部注意力的层位与固定前缀

从同一标准孢子检查点出发，固定全部权重和个体状态。先保持七个全局层，交换第 4、16 层和第 5、17 层的窗口设置，让两处神经记忆读取层采用全局注意力。再比较只限制末端四层或八层的布局。沿用原填充文本，另外拼接未用于本轮调试的验证和测试短文形成第二组填充。同一组内各布局使用完全相同的输入。

| 256 词元窗口方案 | 512 词元键值张量 | 第一组受控记忆 | 第二组受控记忆 | 找回段首站名 |
|---|---:|---:|---:|---:|
| 全局基线 | 14.00 MiB | 5/8 | 7/8 | 2/2 |
| 七个均匀分布的全局层 | 8.73 MiB | 3/8 | 6/8 | 0/2 |
| 七个全局层含第 5、17 层 | 8.73 MiB | 3/8 | 2/8 | 0/2 |
| 末端八层局部 | 11.99 MiB | 5/8 | 4/8 | 0/2 |
| 末端四层局部 | 13.00 MiB | 5/8 | 6/8 | 0/2 |
| 末端四层局部，保留最初 4 词元，仅掩码 | 14.00 MiB | 5/8 | 7/8 | 0/2 |
| 末端四层局部，保留最初 16 词元，实际裁剪 | 13.00 MiB | 5/8 | 7/8 | 2/2 |

将全局范围移到读取层以后，第二组少答对四条。只限制末端四层影响较小，但仍丢掉两道段首题。借鉴 StreamingLLM 保留序列开头的思路，在末端四层试验固定前缀与最近词元共用 256 词元预算。开头 4 个词元主要覆盖提示模板，开头 16 个词元包含两道早期站名题的关键内容。实际裁剪后，末端四层各有 255 个历史键值，前 24 层各为 512 个。相对全局基线下降约 7.2%。键值张量的缩减与本次约 4.94 GiB 的峰值 CUDA 分配是两个指标。 [StreamingLLM](https://arxiv.org/abs/2309.17453)

裁剪版与同样保留 16 个首词元的全存储掩码版，逐词元复现所有站名、短问答和 16 条受控记忆生成。这 16 条受控读取提供地址及版本。两种填充分别约 504–507 和 392–395 词元，只覆盖两个人物。站名放在第二组末端时，保留前缀版与全局基线都答错，说明段首测试的 2/2 无法代表任意位置的长文检索。第 256、257 和 512 词元的分段续算均复现整段运行的首选词元。

小型 CPU 对照验证多段输入的连续位置、受限层的存储上限、全局层的完整存储和临时布局退出后的恢复。共 7 项注意力测试通过。实际裁剪在隔离的本机推理入口运行，支持单序列追加式生成。权重摘要与神经记忆状态摘要通过核对。四轮分别用时约 155、157、154、128 秒。当前候选仍需覆盖落在前缀之外的早期事实、真实多轮对话与自然寻址，并验证标准检查点重载。

## 改变事实位置与前缀长度

在两段填充文本中分别植入一条新接驳车站名，逐次移到开头、第 32 与第 96 个填充词元之后、末段前 32 个词元处。每种布局在同一条输入上对照。生成上限仍为 12 词元，站名出现在可见回答里计为找到。全局基线两轮均为 5/8，逐题通过位置一致。

| 末端四层的前缀长度 | 8 条位置题 | 第一组受控记忆 | 第二组受控记忆 |
|---|---:|---:|---:|
| 全局基线 | 5/8 | 5/8 | 7/8 |
| 16 词元 | 4/8 | 5/8 | 7/8 |
| 64 词元 | 5/8 | 6/8 | 7/8 |
| 128 词元 | 5/8 | 5/8 | 7/8 |

16 词元版丢掉了全局基线在两个第 32 位置找对的站名，并在第一段末端多答对一条。64 和 128 词元版逐题复现基线的正误，两版在第 96 位置的题上都失败。三种前缀都与最近词元共用同样的 256 词元预算，因此末端四层的存储尺寸相同。64 词元版还在一条长提示里恢复了第二个人的历史饮品，其余 15 条记忆生成与全局基线逐词元一致。本轮将 64 词元版选作下一阶段的结构候选，按同等存储预算下的逐题表现选择。新的位置题用时约 144 秒，更宽前缀的对照约 183 秒。

## 增加采用窗口的层数

将 64／256 窗口从末端四层扩到末端八层，512 词元的键值张量从约 13.00 MiB 降至 11.99 MiB。八道站名位置题的正误逐题一致。第二组记忆少答对一条，更正前查询内部期望值为「果汁」，生成改成数字 4。各布局自身的分段续算均复现整段运行的下一词元首选。保留四层为当前标准候选。实验约 178 秒，逐题证据在 `runs/spora-local-attention-deeper-v1/result.json`。

随后用一道普通长提示和一道单独的带记忆提示，比较最后十层中选择八层的 45 种布局。按与全局模型的末位置词表分布差异总和选择第 18、19、20、21、22、23、24、26 层。选择阶段没有使用站名题和原有 16 条换值题。实际裁剪到 11.99 MiB 后，两组受控记忆为 6/8 和 7/8，恢复连续八层布局丢掉的答案。八道站名题逐题复现全局的正误。选择和评测总计约 217 秒。结果在 `runs/spora-local-attention-search-eight-v1/result.json`。

把该布局保存为独立检查点 `runs/spora-anchored-native-eight-v1/bootstrap.pt`。所有原有张量逐元素一致。以调用方单独提供问句的长文本开发题测试，八层、全局和独立重载后的 20 条记录逐项相等。12 条问句中的 6 条在两份内部状态下都答对，八条应拒绝项均未读取。候选 SHA-256 为 `c56112947582dc8559672e31891cef6fe188f8a106c06cc63e146fda7128c9c9`，打包和重载分别约 147 秒和 94 秒。四层与八层的长多轮指令能力仍待对照。

另写四个三轮场景，将 365 词元的无关短文放在第一轮，后续每个模型保留自己的生成历史。原始全局、四层、八层的严格短答均为 0/12，自然结束都是 5/12。四层和八层分别有 10/12 和 11/12 轮与全局模型逐词元一致。八层在人物交换首轮给出了目标物品，但仍带解释，并在后续错判归还后的状态。三种布局都能在寄存更正后给出东门，但未按要求只写地点。多轮数据覆盖窄，实验将每轮限制为 24 词元。逐题输出在 `runs/spora-anchored-dialogue-v1/`。

## 把窗口推进到前半部分

在八层布局上按四层一组向前扩展，避开第一层和两个记忆读取层。用普通长提示和一条独立的记忆输入筛选八种候选。十二层版加第 1–4 层，十六层版再加第 8–11 层。

| 布局 | 键值张量 | 两组受控记忆 | 站名位置题 |
|---|---:|---:|---:|
| 全局 | 14.00 MiB | 5/8、7/8 | 5/8 |
| 八层 | 11.99 MiB | 6/8、7/8 | 5/8 |
| 十二层 | 10.99 MiB | 6/8、8/8 | 4/8 |
| 十六层 | 9.98 MiB | 6/8、8/8 | 4/8 |

十二层丢掉第二组第 32 个填充词元之后的事实，十六层还丢掉第二组末段的事实，并答对第一组末段原来错的一题。各布局中，分段和整段计算仍给出相同的下一词元首选。八层作为重载核对过的候选，十二和十六层保留为结构对照。计算约 227 秒，详细结果在 `runs/spora-local-attention-expand-v1/result.json`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$checkpoint = 'runs/spora-native-query-key-v1/bootstrap.pt'
.\.venv\Scripts\python.exe -u scripts/audit_spora_local_attention.py --checkpoint $checkpoint --compare-memory-reads --output runs/spora-local-attention-memory-read-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_local_attention.py --checkpoint $checkpoint --compare-tail --output runs/spora-local-attention-tail-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_local_attention.py --checkpoint $checkpoint --compare-anchors --output runs/spora-local-attention-anchors-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_local_attention.py --checkpoint $checkpoint --compare-bounded-anchors --output runs/spora-local-attention-bounded-anchors-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_local_attention.py --checkpoint $checkpoint --compare-fact-placements --output runs/spora-local-attention-placements-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_local_attention.py --checkpoint $checkpoint --compare-wider-prefix --output runs/spora-local-attention-wider-prefix-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_local_attention.py --checkpoint $checkpoint --compare-deeper --output runs/spora-local-attention-deeper-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_local_attention.py --checkpoint $checkpoint --search-eight --output runs/spora-local-attention-search-eight-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_local_attention.py --checkpoint $checkpoint --expand --output runs/spora-local-attention-expand-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_dialogue.py --checkpoint $checkpoint --output runs/spora-anchored-dialogue-v1/full.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_dialogue.py --checkpoint runs/spora-anchored-native-candidate-v1/bootstrap.pt --output runs/spora-anchored-dialogue-v1/four.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_dialogue.py --checkpoint runs/spora-anchored-native-eight-v1/bootstrap.pt --output runs/spora-anchored-dialogue-v1/eight.json
```
