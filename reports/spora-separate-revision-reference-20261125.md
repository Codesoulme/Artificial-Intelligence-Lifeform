# 分开识别确认动作和草稿指向

此前把草稿描述、前文和末句一起编码，校准后在 96 条明确指向题中没有正确提交。本轮把两个草稿和建立指向的发言分别编码。孢子冻结前 20 层提供特征，引用匹配头比较两个草稿并可选择「无法确定」。另一头独立判断末句是否允许提交。匹配头的候选分数按草稿顺序一起交换，无法确定的分数只依赖当前发言。提交仍用 `RevisionDraft` 写入隔离的工作状态。 [前轮对照](spora-revision-reference-20261125.md) · [Event-DST](https://aclanthology.org/2026.conll-main.30/)

训练把已知字段扩为五种称呼，并加入包含确认词却暂缓、撤回的语句。合计 7,200 个合成决策。阈值仅从 360 个校准决策选取。训练与校准均排除放映通知场景。留出使用未参与训练的措辞，包含前一轮同一批 144 道题，以及增加暂缓和撤回的 864 道题。草稿的字段、描述和完整值由评测给定。

| 评测 | 目标草稿排名正确 | 正确提交 | 错误提交 | 最终状态完全正确 |
| --- | ---: | ---: | ---: | ---: |
| 原 144 题 | 96/96 | 48/96 | 0/144 | 96/144 |
| 扩展 864 题 | 536/576 | 148/288 | 8/864 | 724/864 |
| 扩展集中未参与训练的放映场景 | 96/96 | 32/48 | 0/144 | 128/144 |

在扩展题中，8 条误写全部来自园艺场景的指向选错。动作头在这批留出上接受了所有应提交的确认，没有把暂缓或撤回当作提交。给出正确的引用标签后，用该动作头回放可正确提交 288/288，误写为零。继续换动作词收益有限，更值得补上字段近义表达、跨主题同字段以及真实会话中的候选提取。原 144 题虽然候选相对排序全对，仍有 48 条低于提交边界。当前实验只提供已知草稿，完整自动写入和开放聊天仍需独立验证。

相较前轮，本轮同时改变了结构和训练措辞覆盖，无法把全部改善归给其中一个因素。训练和评测约 92 秒，独立重载约 85 秒，逐题复现分数和状态回放。两次 32 条受控持久记忆读回均与父实验逐条一致，个体摘要一致。15 项相关单元测试通过，另核对了与前轮相同的 144 条决策及草稿交换后的分数行为。逐行报告位于 `runs/spora-separate-revision-reference-v1/result.json`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$sporaSplit = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
$sporaArgs = @(
    'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
    'runs/spora-revision-reference-clause-v1/result.json',
    'runs/spora-separate-revision-reference-v1',
    "$sporaSplit/layer-19", "$sporaSplit/layer-22"
)
.\.venv\Scripts\python.exe -u -m scripts.train_spora_separate_revision_reference @sporaArgs
.\.venv\Scripts\python.exe -u -m scripts.train_spora_separate_revision_reference @sporaArgs --verify
```
