# 同深度的循环注意力对照

上一轮比较前 12 层与前 23 层，读出位置和经过的注意力一起变了。本轮用同一个融合父检查点，分别保留其注意力和替换索引 19、22 两处注意力。两组都用前 23 层特征、相同的逐条无填充提取、随机初始化规则和 144 条训练视图。字段排序和目录核验继续使用同一个模型。前 12 层作为负对照，保持两个替换位置都在读出之后。实验设计参考混合 xLSTM 的匹配下游对照。 [混合 xLSTM 蒸馏](https://proceedings.mlr.press/v306/hauzenberger26a.html) · [深度实验](spora-lookup-intent-depth-20261126.md)

| 前 23 层读出的指标 | 父检查点 | 双循环层 |
| --- | ---: | ---: |
| 独立校准的单项正确提议 | 16/24 | 16/24 |
| 留出单项正确提议 | 8/32 | 8/32 |
| 留出单项错提议 | 2/32 | 2/32 |
| 留出零项误提议 | 0/16 | 0/16 |
| 留出双项误提议为单项 | 4/16 | 4/16 |
| 留出两个分句均判对 | 38/64 | 38/64 |

两组都使用校准选出的 0.99。在 168 条校准和评测视图中，前 12 层的保存分数与训练张量逐项一致。前 23 层训练张量的最大绝对差为 0.00771，保存的分句概率最大差为 0.00364。阈值后的分句判定、单项选择和记录提议一致。这个结果仅覆盖短分句和当前探针，尚未检验长对话中循环状态对查阅的影响。

事后检查阈值时，循环版前 23 层用 0.90 可把新留出中两个分句均判对从 38/64 提到 45/64，零项无误报，仍有 7/16 双项没识别完整。同一阈值在校准集使零项误报增至 5/12。这个回看用于定位校准与留出分布差异，0.90 未作为发布阈值。下一轮需要按不同问法家族评估执行者、引用和推迟这三种语义，并用新的测试题比较把整句话一起编码是否能改善双项判断。 [神经网络校准](https://arxiv.org/abs/1706.04599)

父检查点对照约 117 秒，输入摘要检查通过，35 项相关测试通过。父组结果位于 `runs/spora-lookup-intent-depth-parent-v1/result.json`，对应循环组位于 `runs/spora-lookup-intent-depth-v2/result.json`。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_intent_depth runs/spora-switch-address-matched-v1 runs/spora-lookup-field-focus-independent-v1 runs/spora-lookup-schema-ranking-v1 runs/spora-lookup-intent-depth-parent-new --recurrent @weights --attention parent
```
