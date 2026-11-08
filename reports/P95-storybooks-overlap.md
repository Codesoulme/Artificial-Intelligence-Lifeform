# Storybooks Canada 候选去重与历史来源诊断

## 结果

- 29 篇候选全部保留：质量评审标记为通过 20 篇、存疑 9 篇；所有条目的训练批准状态均为否。该标记原样记录，不代表本次批准。
- 逐项核对了 `page-text-verification-v2.json`、两份质量评审与原始文件的 SHA-256，29 项一致；两份评审各覆盖 15 和 14 篇，合并后无缺漏或重复。核验文件记录的页面正文匹配和许可检查均为真；本次没有重新联网核实权利或网页。
- 29 篇互相比较：归一化完整正文、12 字符和 20 字符片段均无重合。
- 已知祖先 `runs/conversation-data-v5/pretrain-train.jsonl`：扫描 1,085,328 条，有 1 篇候选的 1 个 12 字符位置在 1 条记录出现 1 次；没有完整正文包含或 20 字符重合。
- P28 `runs/p28-stories-data/candidate-v3/pretrain-train.jsonl`：扫描 100,000 条，有另一篇候选的 1 个 12 字符位置在 1 条记录出现 1 次；没有完整正文包含或 20 字符重合。P28 pilot 配置指向 candidate-v3，清单标明该文件为故事训练集。

原始逐篇结果位于本地 `runs/storybooks-dedup-diagnostic-20261108/results.json`，使用原文件与正文的 SHA-256 区分条目，包含原评审标记、计数与范围，不含正文、标题或来源记录文本。输入文件和扫描来源的 SHA-256 也记录其中。

## 方法和边界

根据核验记录的页数，从原始 Markdown 取正文页，排除标题和末尾署名；按 Unicode NFKC、casefold 处理后移除非字母数字字符，再连接各页。以 Aho-Corasick 同时查找完整正文及所有滑动 12、20 字符窗口，每条训练记录独立扫描，窗口不会跨记录。完整包含包括训练记录更长的情形；完整相等另行计数。候选互比时排除自身。

`candidate_ranges` 是归一化正文中的零起点、右端不含的字符区间，表示有重合的窗口在候选侧的并集；它不是原始文件字节偏移。`occurrences` 统计候选窗口位置与来源中出现位置的配对次数，窗口可相互重叠；`documents` 是至少出现一次的来源记录数，`candidate_window_positions` 是候选侧不同窗口起点数。全零也明确保留。归一化会跨页、标点和空白拼接，可能造成额外重合。

常见短句或偶然一致可能导致短片段误报，12 字符结果不足以证明作品重复。此方法不能检测翻译、改写和语义近重复，也不能证明模型是否见过某篇内容；历史训练还可能涉及本次未扫描的其他来源，且数据文件存在不证明其每条记录实际被采样。无重合只适用于列出的文件、提取方式和阈值。本次没有读取最终测试正文，没有修改训练文件或 `spora`，也没有训练模型。

## 复现

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_storybooks_overlap.py -v
.\.venv\Scripts\python.exe -B scripts/storybooks_overlap.py
```

脚本依赖 `requirements-data-audit.txt` 中的 `pyahocorasick`，输入路径与原诊断相同。第二条命令会重新扫描上述两份大型训练数据，只有需要复现审计时才执行；本次迁移没有执行该命令。扫描和校验成功后，脚本在 `runs/` 下创建一个全新的 `storybooks-overlap-` 前缀目录，并以独占创建方式写入 `results.json`，不会覆盖已有结果。校验或扫描失败时不会创建结果目录。原始本地报告和结果保持不变。
