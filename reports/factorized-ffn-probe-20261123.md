# 前馈投影低秩结构探针

日期：2026-11-23。

## 目的

在 DeepSeek-R1-Distill-Qwen-1.5B 的全层 Qwen 兼容孢子骨架上，把第 6 层前馈输出投影改成低秩双线性映射，观察一种真正改变内部参数化、但不替换整层注意力和位置协议的迁移方式。

原始层、分词器、孢子个体和持久状态均保持不变。探针只写入 `runs/spora-factorized-down-layer6-r256-v5/`。

## 方法

1. 收集第 6 层原始前馈输出投影在中文对话序列上的真实输入激活。
2. 用固定随机低秩输入投影和岭回归拟合原投影输出。
3. 将低秩因子保持为 FP32，前馈结果回到骨干精度。
4. 冻结底座，只用原模型的中文教师分布、目标文本和当前／历史记忆回放训练 32 步。
5. 在独立进程中重新加载低秩因子并复测。

这条实现参考了函数保持式网络扩展、激活感知低秩拟合以及近期低秩状态表示研究的共同原则：先约束已有函数，再逐步改变参数化。它不是把原始权重直接截断后宣布完成迁移。

## 结果

来源：`runs/spora-factorized-down-layer6-r256-v5/result.json` 和 `reload.json`。

| 条件 | C-Eval | 多轮首轮 | 多轮追问 | 两轮均正确 | 记忆正确 |
|---|---:|---:|---:|---:|---:|
| 兼容骨架 | 24/60 | 2/8 | 1/8 | 1/8 | 7/8 |
| 秩 256，未训练 | 24/60 | 1/8 | 1/8 | 1/8 | 3/8 |
| 秩 256，教师训练 32 步 | 23/60 | 1/8 | 1/8 | 1/8 | 3/8 |

因子重载后的结果与训练进程内完全一致：`23/60`、`1/8`、`1/8`、`1/8`、`3/8`。个体状态摘要保持不变。原投影矩阵的相对重构误差为 `1.2222`，说明当前随机输入投影加回归没有覆盖足够的语言表示方向。

秩 512 对照在完成教师训练前后未生成结果目录，进程被本机资源回收，不能视为模型指标，也不作为失败性能结论。

## 决定

当前不把低秩完整前馈替换接入正式检查点。它证明了：

- 内部参数化替换可以在缓存协议不变的情况下运行；
- 只拟合局部输入激活不足以保持跨状态记忆读取；
- 教师分布训练 32 步无法修复结构初始误差；
- 低秩秩数提高会显著增加本机内存和运行时间。

下一阶段从 `runs/qwen-compatible-memory-gated-residual-v1/` 的路线继续：保留 Qwen 的注意力、归一化、位置编码、前馈和 KV 状态，只把孢子路径作为零初始化、记忆读取时才启用的内部残差。它在独立人物留出上达到 `17/24`，同题换值 `6/12`，普通 C-Eval 与双轮基线逐题保持；这些数字仍不是自由聊天验收，但比直接替换完整前馈更适合作为下一迁移单元。

## 复现

```powershell
.venv/Scripts/python.exe scripts/research_factorized_qwen_ffn.py --ceval runs/ceval-validation-617524a --output runs/spora-factorized-down-layer6-r256-v5 --layer 6 --rank 256 --steps 32 --learning-rate 1e-5
.venv/Scripts/python.exe scripts/verify_factorized_qwen_ffn.py --source runs/qwen-compatible-raw-v1/integrated.pt --candidate runs/spora-factorized-down-layer6-r256-v5/factorized-down.pt --ceval runs/ceval-validation-617524a --output runs/spora-factorized-down-layer6-r256-v5/reload.json --layer 6 --rank 256
```

## 在较强中文对话候选上复测

以 `runs/named-memory-dialogue-midlayers-composition-v1/integrated.pt` 为起点，替换第 6 层的前馈输出映射。两种初始化共用同一批 4,096 个前馈激活位置，前五分之四用于拟合，末五分之一留作投影误差对照。随机岭回归学习固定随机输入投影的输出。另一组从原始输出中提取主方向，将稠密映射投影到这些方向。它考察激活分布对低秩近似的影响，公式与 SVD-LLM 的白化和逐层更新流程有差异。 [SVD-LLM](https://arxiv.org/abs/2403.07378)

| 条件 | 留出投影相对误差 | C-Eval | 完整两轮 | 受控记忆 |
|---|---:|---:|---:|---:|
| 原始输出映射 | — | 21/60 | 4/8 | 8/8 |
| 随机岭回归，秩 256 | 0.971 | 24/60 | 2/8 | 8/8 |
| 输出主方向，秩 256 | 0.682 | 22/60 | 2/8 | 8/8 |
| 输出主方向，秩 512 | 0.570 | 18/60 | 1/8 | 8/8 |
| 输出主方向，秩 256，再训练 32 步 | 0.682（训练前） | 22/60 | 2/8 | 8/8 |

32 步训练只更新两块低秩矩阵，使用保持项目对话和受控记忆预测的教师分布。独立重载复现全部公开题选项、短对话和记忆结果，个体状态摘要一致。投影误差下降与多轮正确率回升没有同时发生。四个替换候选均低于原始输出映射的多轮成绩，本轮保留原映射。

```powershell
$sporaSource = 'runs/named-memory-dialogue-midlayers-composition-v1/integrated.pt'
$sporaRun = 'runs/spora-dialogue-ffn-layer6-output-subspace-r256-distill-v1'
.\.venv\Scripts\python.exe scripts/research_factorized_qwen_ffn.py --source $sporaSource --ceval runs/ceval-validation-617524a --output $sporaRun --layer 6 --rank 256 --steps 32 --learning-rate 1e-5 --initializer output-subspace
.\.venv\Scripts\python.exe scripts/verify_factorized_qwen_ffn.py --source $sporaSource --candidate "$sporaRun/factorized-down.pt" --ceval runs/ceval-validation-617524a --output "$sporaRun/reload.json" --layer 6 --rank 256
```
