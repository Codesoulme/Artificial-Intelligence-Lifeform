# 第二头组的整网多轮对齐

在第 18 层第二头组的 8 对绝对位置坐标上，先复现 96 步局部对齐，再做 96 步整网对齐。新增投影的权重和偏置是唯一参与优化的参数，位置表、第一头组、语言骨干和个体状态均固定。训练用 41 轮标注对话、16 段短文和 20 条带内部记忆的回放。对话同时匹配父模型预测和标注答案，短文与记忆回放匹配父模型预测。11 轮对话、4 条回放用于这个训练阶段的留出检查。开发和独立生成评测的问句均排除了与对话训练的逐字重合。

## 结果

| 指标 | 局部对齐后 | 再加整网对齐 |
|---|---:|---:|
| 留出对话的标注答案 NLL | 1.854 | 1.837 |
| 留出对话的父模型 KL | 0.000261 | 0.000241 |
| 留出记忆回放的父模型 KL | 0.000272 | 0.000404 |
| 原 60 道 C-Eval 答对数 | 25/60 | 25/60 |
| 原 60 道 C-Eval 选项与父模型一致 | 60/60 | 60/60 |
| 无蛋早餐追问 | 列入鸡蛋 | 列入鸡蛋 |

另用四个新写场景各跑两轮，双方复用相同的贪心解码和 64 词元上限，历史使用自己的实际回复。新候选的 8 轮生成中，6 轮与父模型逐词元一致。双方都定位到更正后的取件地点，但回复比要求更长。双方也都算错累计数量的追问，并在改为晚安问候时重复了早安。剩余两轮是菜单场景。首轮父模型确实推荐了青菜炒饭，候选只复述用户问题。用户更正只剩米饭后，父模型仍建议青菜，候选则要求用户再提供主料。这个结果暴露了生成质量的变化，选择题和固定前缀损失都没反映出该问题。

整网候选保留为绑定父检查点的实验投影。无蛋开发用例仍退化，候选没有进入标准检查点流程，也没有运行正式的 32 条受控记忆换值。已核对投影文件摘要。有效运行约 204 秒，峰值 CUDA 分配 4.94 GiB。首次在收集教师回放时因两个调用路径重复传入生成状态参数而中止，修正参数合并后完成。6 项位置注意力测试和双头组增量重载测试通过。逐题输出与权重在 `runs/spora-query-key-dialogue-alignment-v1/`。

下一轮可在同一坐标范围比较基于学生实际输出的蒸馏。先用候选生成训练前缀，再让冻结父模型在相同前缀给出后续分布，用独立多轮生成和受控记忆换值判断效果。GKD 在其他任务上研究了这类训练与部署之间的序列分布偏差，该研究结果尚未在此结构上复现。 [GKD](https://arxiv.org/abs/2306.13649)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -u scripts/audit_spora_query_key_expansion.py --checkpoint runs/spora-native-query-key-v1/bootstrap.pt --reference runs/spora-partial-query-key-v1/verified.json --attention runs/spora-grouped-absolute-attention-v1/attention.pt --ceval runs/ceval-validation-617524a --second-group-pairs 8 --calibration-steps 96 --dialogue-steps 96 --output runs/spora-query-key-dialogue-alignment-v1/result.json
```
