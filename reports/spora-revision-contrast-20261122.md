# 自动记忆查询的更正前后对照

从已经纳入部分查询与键位置参数的标准检查点出发。保留姓名定位器、读入意图参数和记忆回答层，训练一个只修正当前、改前、改后三类分数的小模块。在已有成组三问中加入新的对照句，让「不要改前」「改前还没动笔」「改完后」等语义出现在相近句式中。

两份候选分别使用冻结骨干的语义特征，以及语义加字符一至三元组特征。用不同人物和句式的校准集选择残差权重与拒读阈值。随后在两组已有留出及另写的句式上检查时间分类。端到端复验使用 72 条旧版口语正例、40 条应拒读的旧负例，以及 96 条新正例和 32 条新负例。每道正例要在两个内部状态下自行选对人物和版本，两份答案都正确才算通过。数据覆盖模拟人物的两份饮品记录。

## 全类别残差

408 条三类正例用于拟合，76 条独立问句用于校准。校准选择了仅用语义特征的候选，读入阈值从 0.541 升到 0.806。

| 端到端评测 | 原版自动换值 | 残差自动换值 | 改前问答通过 | 负例触发读取 |
|---|---:|---:|---:|---:|
| 旧留出 | 34/72 | 37/72 | 9 → 12 | 0/40 → 0/40 |
| 新留出 | 46/96 | 38/96 | 10 → 13 | 2/32 → 0/32 |

这一设置提高了历史问答，但新留出中时态正确数由 58 降到 43，自动换值总数也下降。保留原有读取和拒读决定，再测试只把可信的误判纠正为「改前」的方案。该方案从同一残差权重出发，只在校准集选择修正强度和最低置信度。

## 选择性纠错

沿用原模型的时态分类和读入决定。仅在原模型已经选择读取当前或改后、残差给「改前」的条件概率达到 0.55 且至少比原分数高 0.15 时，改成读取历史。0.55 和残差系数 1.25 由 76 条校准题选出。

| 端到端评测 | 自动换值 | 改前问答通过 | 负例触发读取 |
|---|---:|---:|---:|
| 旧留出 | 34/72 → 37/72 | 9/24 → 12/24 | 0/40 → 0/40 |
| 新留出 | 46/96 → 61/96 | 10/32 → 25/32 | 2/32 → 2/32 |

分别新增 3 条和 15 条正确的历史回答，原本通过的问句都保住了。240 条的读取和拒读决定均与原版相同。新留出的两条误读来自「封存某人的旧记录，只介绍饮品这个词」的句式。候选配合当前标准检查点试验，数据覆盖人物和饮品的两份记录。更多版本和真实日期需另做评测。

重新启动进程加载保存的 88,844 字节候选后，逐项复现了 240 条问句的路由与回答。候选 SHA-256 为 `37fe94e0c2782e81f8cdfb1cc763b9d5b7b250f61b150daa1a504db1aabeacd9`。拟合及全类别评测约 343 秒，选择性校准与评测约 230 秒，重载复验约 227 秒。27 项相关测试通过。逐题记录在 `runs/spora-past-override-v1/`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe scripts/train_spora_revision_contrast.py --checkpoint runs/spora-native-query-key-v1/bootstrap.pt --route-source runs/qwen-compatible-gelu-layer17-v1/integrated.pt --router runs/qwen-compatible-gelu-lexical-boundary-v1/hybrid --temporal runs/qwen-compatible-gelu-temporal-factorized-v1/factorized --output runs/spora-revision-contrast-v1
.\.venv\Scripts\python.exe scripts/audit_spora_past_override.py --checkpoint runs/spora-native-query-key-v1/bootstrap.pt --route-source runs/qwen-compatible-gelu-layer17-v1/integrated.pt --router runs/qwen-compatible-gelu-lexical-boundary-v1/hybrid --temporal runs/qwen-compatible-gelu-temporal-factorized-v1/factorized --residual runs/spora-revision-contrast-v1 --output runs/spora-past-override-v1
.\.venv\Scripts\python.exe scripts/audit_spora_past_override.py --checkpoint runs/spora-native-query-key-v1/bootstrap.pt --route-source runs/qwen-compatible-gelu-layer17-v1/integrated.pt --router runs/qwen-compatible-gelu-lexical-boundary-v1/hybrid --temporal runs/qwen-compatible-gelu-temporal-factorized-v1/factorized --residual runs/spora-revision-contrast-v1 --output runs/spora-past-override-v1 --reopen
```
