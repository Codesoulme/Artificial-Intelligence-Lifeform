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
```
