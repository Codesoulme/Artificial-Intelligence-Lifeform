# 八层局部注意力候选的中文对话适配

在已经通过独立重载的八层局部注意力检查点上，于第 24、27 层增加零初始化秩 16 的孢子残差，共 98,304 个可训练参数。固定底座权重、窗口设置、记忆读取器与个体。训练用 189 条短标注回答及对应的 189 条长输入。长输入在首条用户消息前加约 290 词元短文，练习从长输入里识别问句并延续对话。另有 24 条记忆回放，及从短对话中选出的 24 条父模型分布回放。

## 首轮结果

三种模型使用相同的贪心解码、关闭思考段、每轮最多 48 词元。四个短场景来自已有开发集。四个长场景来自窗口审计，每个模型把自己实际生成的回答传入后续轮次。已排除逐字相同的训练问句，场景仍属于开发诊断。

| 配置 | 长场景严格短答 | 24 轮自然结束 | 三科 C-Eval | 受控记忆精确换值 |
|---|---:|---:|---:|---:|
| 原版 DeepSeek | 0/12 | 19/24 | — | — |
| 八层孢子父模型 | 0/12 | 22/24 | 25/60 | 32/32 |
| 训练 192 步 | 1/12 | 21/24 | 25/60 | 32/32 |

候选保留了 60 道选择题的全部原选项和 32 条记忆结果。记忆对照使用已指定的人物和版本。短场景中，候选能回答上午去了邮局、接受改约，并在听到下雨后选择在家看电影。它也会复述第一轮的邮局答案，在最后问起约会安排时拒答。长场景中，首轮能精确输出「西门」，之后两轮在用户已经改为「东门」的情况下仍回答西门。父模型两次都能给出新地点，但违反只写地点的要求。日期和格式场景仍然失败。选择题和有限记忆保持的证据不能覆盖多轮纠正。

训练与评测约 366 秒，峰值 CUDA 分配 4.94 GiB。候选在 `runs/spora-anchored-dialogue-residual-v1/candidate.pt`，SHA-256 为 `af9f679e934097588c50a77e38d4ced02a77911ef671ee6c620d8ee4229c2d03`。独立进程确认个体摘要、窗口布局，以及 24 轮回复、60 道选择题和 32 条记忆生成全部复现。逐题记录在该轮的 `result.json`，重载记录在 `runs/spora-anchored-dialogue-residual-v1-reload/result.json`。

## 使用自身回复续训

从已重载验证的第一轮候选出发。在八个训练场景采集模型实际产生的回复，分别用短输入和含填充的长输入展开，形成 32 条追问训练样本。追问的监督答案来自原标注。混合第一轮的短答、长答与保持目标，继续训练 128 步。

严格短答仍是 1/12，自然结束仍为 21/24。模型开始在对调行程后答「菜市场」，也能给出人物偏好的「温水」，但错误地把该值复用到询问原因的追问。长场景的地点更正仍失败，日期题出现明显机械重复。C-Eval 由 25/60 下降到 23/60，32 条受控记忆逐项复现。续训与评测约 342 秒，峰值 CUDA 分配 4.94 GiB。逐题和候选保存于 `runs/spora-anchored-dialogue-residual-v2/`。两轮均保留作研究，八层父模型仍作为结构迁移的起点。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$evalArgs = @('--ceval', 'runs/ceval-validation-617524a')
.\.venv\Scripts\python.exe -u scripts/train_spora_anchored_dialogue.py @evalArgs --checkpoint runs/spora-anchored-native-eight-v1/bootstrap.pt --steps 192 --output runs/spora-anchored-dialogue-residual-v1
.\.venv\Scripts\python.exe -u scripts/train_spora_anchored_dialogue.py @evalArgs --checkpoint runs/spora-anchored-dialogue-residual-v1/candidate.pt --verify-report runs/spora-anchored-dialogue-residual-v1/result.json --output runs/spora-anchored-dialogue-residual-v1-reload
.\.venv\Scripts\python.exe -u scripts/train_spora_anchored_dialogue.py @evalArgs --checkpoint runs/spora-anchored-dialogue-residual-v1/candidate.pt --steps 128 --on-policy --output runs/spora-anchored-dialogue-residual-v2
```
