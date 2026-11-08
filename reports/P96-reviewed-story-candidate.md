# 中文故事候选：整理与复核

状态：仅供本地审查，**未批准训练**。首轮通过 20 篇，其中独立复审通过 19 篇；候选包含 15 篇训练组、4 篇开发组，合计 223 段、10,500 个正文字符、8,851 个正文词元。组别仅为候选划分，不构成使用许可。

脚本位于 `scripts/build_storybooks_zh_candidate.py`，合成测试位于 `tests/test_build_storybooks_zh_candidate.py`。在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_build_storybooks_zh_candidate.py -v
.\.venv\Scripts\python.exe -B scripts/build_storybooks_zh_candidate.py --output-name <runs下尚不存在的目录名>
```

需要保留原研究素材、审查记录、历史审计和既有词表，并在项目虚拟环境中安装 `tokenizers`。输出名称必须是 `runs/` 下的单级新目录；已存在的目录会被拒绝，文件以独占方式创建。写入阶段若中断，可能留下不完整的新目录；检查后应选择另一个新名称。`runs/` 不纳入 Git。

本次 7 项合成测试通过；用新目录 `runs/storybooks-zh-p96-reproduction-20261108/` 重建后，与既有 19 篇结果的三个文件逐字节一致。再次使用该目录名被拒绝。文件 SHA-256：

- `train.jsonl`：`ad104d6908b326e97ec55c13c3870080a99837ee69e70b1e5621e86b344ec337`
- `dev.jsonl`：`5889a41e22ca42be6952177f17ae8c1ba727df868668d94843ff8c1cb36d6b11`
- `manifest.json`：`b86db1fb4bdfa2519fc87ecbd8ed7f91b0ba09b01c7348053bfef5cf1a2cc60b`

网页快照记录的许可版本为 CC BY 3.0 共 13 篇、CC BY 4.0 共 6 篇；上游权利、各方授权、署名及实际使用条件仍需逐篇人工复核。模型审查不等于人工认证，也未对照外文原作验证翻译。历史重合结论沿用既有审计，只覆盖其中列明的两个来源；本次没有重新扫描这些大文件，亦不能排除其他来源、改写或语义近重复。样本规模不足以支持训练效果或泛化结论。本次未读取最终测试正文、修改既有数据或模型，也未启动训练。
