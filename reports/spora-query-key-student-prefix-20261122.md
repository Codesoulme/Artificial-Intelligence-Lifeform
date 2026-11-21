# 第二头组的学生回复前缀蒸馏

从上轮已经过局部对齐和整网对齐的 8 对坐标候选续训。校验候选投影和父检查点的摘要，冻结父模型、第一头组、绝对位置表与个体，只更新第二组投影。六个训练场景由候选自行贪心生成，每轮最多 20 个词元，实际回复进入下一轮历史。18 轮共 341 个生成词元。冻结父模型对同一批前缀给出后续分布，第 48 步重新生成，4 轮的输入或输出词元发生变化。96 步中，36 步用于这些生成轨迹，其余为标注对话、短文和带状态的记忆回放。开发与额外复测问句均排除逐字重合。方法参考了 GKD 对学生生成分布的训练思路。 [GKD](https://arxiv.org/abs/2306.13649)

## 结果

| 指标 | 续训前 | 续训后 |
|---|---:|---:|
| 初始学生轨迹的父模型 KL | 0.000144 | 0.000133 |
| 11 轮标注留出对话的答案 NLL | 1.8370 | 1.8301 |
| 11 轮标注留出对话的父模型 KL | 0.000241 | 0.000361 |
| 4 条留出记忆回放的父模型 KL | 0.000404 | 0.000292 |

完成全部运行评测后，143 道选择题的选项与冻结父模型逐题一致，两个分组分别答对 `25/60`、`23/83`。24 轮开发对话中 19 轮和父模型逐词元一致。32 条受控当前／历史答案逐项一致，四条问句均随八份内部状态正确换值。普通生成和带受控记忆生成各一条键值复用对照通过。以上换值评测提供地址和时间版本。

无蛋早餐追问仍加入鸡蛋。上轮四个额外场景的 8 轮回答与续训前逐词元一致，菜单首轮仍只复述提示，取件场景仍能找到更新后的地点但未遵守简短格式。固定前缀的数值变化未转化到这批多轮生成。继续使用单头组标准检查点，续训后的投影单独保留。完整结果与通过摘要核对的投影在 `runs/spora-query-key-student-prefix-v1/`。全程约 384 秒，峰值 CUDA 分配为 4.94 GiB。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -u scripts/audit_spora_query_key_expansion.py --checkpoint runs/spora-native-query-key-v1/bootstrap.pt --reference runs/spora-partial-query-key-v1/verified.json --attention runs/spora-grouped-absolute-attention-v1/attention.pt --second-group-pairs 8 --ceval runs/ceval-validation-617524a --resume-component runs/spora-query-key-dialogue-alignment-v1/result.pt --dialogue-steps 96 --student-prefixes --output runs/spora-query-key-student-prefix-v1/result.json
```
