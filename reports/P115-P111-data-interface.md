# P115 候选接入 P111

P115 只生成候选故事流，不能直接替代 P111 的完整数据目录。P111 的 `data`、`base_data`、课程、监测数据、既有审查和审计仍按原配置提供；此接入只替换 `story` 来源。没有创建训练计划、批准书或审计结果。

后续准备时，可在新的 P111 计划中加入：

```json
"candidate_story": {
  "path": "runs/新的候选目录",
  "summary_sha256": "该目录 summary.json 的完整 SHA256"
}
```

必须启用 `story_sampling_mode: "uniform"` 和故事文档边界及覆盖记录；窗口至少容纳最长故事的内容词元数加一个 EOS 目标。其分词器摘要必须与 `data/manifest.json` 一致。校验会复算记录身份、全文编码、偏移、二进制、统计、拒绝记录和产物摘要，拒绝正文包含 BOS、EOS 或 PAD，以及超出模型词表或窗口的词元。整个故事文件必须非空。

`training_source_reviews.story` 仍需指向独立批准记录，满足原有范围、步数等检查，并同时包含 `training_source_sha256.story`（实际 `stories.bin` 的 SHA256）和 `candidate_summary_sha256`（实际清单的 SHA256）。旧数据的批准不能因字段沿用而自动覆盖该候选。旧数据的审计也不证明新故事与保留集合隔离；新数据的来源、授权、内容和隔离审查仍需另行完成。此接入不会生成或冒充这些证明。

清单与各文件在验证前后复核，训练段结束时再检查；验证与读取之间并非原子快照，运行时应确保候选目录不可更改。若训练期间检测到变更，已生成的中间检查点不能视为有效结果。P111 的实现指纹已变化，旧版续跑检查点不兼容，不应声称可以原样续跑。

验证命令（只使用合成故事及本地分词器，不启动训练）：

```powershell
cd D:\SPORA
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
.venv/Scripts/python.exe -B -m unittest tests.test_p115_p111_input tests.test_source_review tests.test_p111_chunk_coverage -v
```
