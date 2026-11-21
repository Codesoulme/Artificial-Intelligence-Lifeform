# 局部注意力下的自然记忆查询

比较全局检查点与末端四层采用 64 个固定前缀词元、256 词元总窗口的候选。两个条件使用相同的神经记忆、身份定位器和时间分类器。四个人各有当前、旧值、更正后查询，另有八条不应触发读取的问句。每次读取在两个不同内部状态上生成，正确要求自动选中适当人物和时间，再按状态给出不同的正确值。

| 输入与定位方式 | 布局 | 人物正确 | 时间正确 | 两状态答对 | 不应读取却触发 |
|---|---|---:|---:|---:|---:|
| 短问句 | 全局与局部 | 8/12 | 9/12 | 6/12 | 0/8 |
| 长文本整体参与定位 | 全局与局部 | 1/12 | 4/12 | 0/12 | 8/8 |
| 调用方单独提供当前问句，生成保留完整长文本 | 全局与局部 | 8/12 | 9/12 | 2/12 | 0/8 |

第一轮全局分支沿用旧测试的无键值存储设置，裁剪分支启用了存储，短问句分别答对 6/12、5/12。短序列直接使用常规掩码后，这个差异仍在。显式将两个条件都设为启用键值存储后，短问句和长文本的全部逐题记录一致。此前短问句的差异来自实验条件。7 项注意力测试通过，包括短序列输出相同和裁剪续算。

长文本中的干扰人名劫持了两种布局共用的记忆定位。单独提供当前问句后，八条应拒绝项全部恢复，回答仍看见完整长文本。六条人物和版本都对的长问句里，四条至少有一份状态的模型输出未给出矩阵读到的正确值。此对照以调用方提供的问句边界为前提。单条长消息中的自动分段和跨轮指代尚需验证。

同口径对照约 204 秒，单独提供问句的对照约 113 秒。逐题结果在 `runs/spora-anchored-natural-memory-v3/result.json` 和 `runs/spora-anchored-natural-memory-separated-v1/result.json`。

## 输出和检查点验证

单独提供问句后，已有的神经记忆输出偏置从 1 倍调到 1.5 倍，两种布局均从 2/12 升到 6/12，六条分别在两个状态下精确输出神经记忆值。2 倍也为 6/12。另选四个人、不同句式和不同填充，1.5 倍与原设定均为 0/12，定位人物有 10/12，但两个布局的时间版本均为 0/12。八条应拒绝项均未读取。两轮分别约 156 秒和 186 秒。第一批输出实验支持 1.5 倍作为开发选项，第二批显示时间选择仍限制整体表现。逐题结果在 `runs/spora-anchored-natural-memory-gains-v1/` 与 `runs/spora-anchored-natural-memory-heldout-v1/`。

在独立重载的窗口检查点上换用已有的选择性「改前」纠正头，留出结果仍为时间 `0/12`、正确换值 `0/12`、负例误读 `0/8`。三类错误都发生在选择记录版本时，其中「照有效条目」多被判为更正后，「换掉的那项」多被判为更正后，「第一次修订生效」多被判为更正前。逐题见 `runs/spora-anchored-natural-memory-heldout-override-v1/result.json`。

已把 64／256 的末端四层配置写入独立标准检查点。保存前后所有张量逐元素相等，元数据只新增窗口配置。通过标准运行入口加载后，带长文本的 20 条开发记录全部复现，包括 6/12 组正确换值与 0/8 应拒绝却触发。小型模型回归还验证了增量写入后重启、局部层键值上限和逐词元生成一致性。实机打包与独立重载分别约 150 和 102 秒。候选在 `runs/spora-anchored-native-candidate-v1/bootstrap.pt`，SHA-256 为 `87b8eb46e5a46f5e0dd338c350faf8b9932bf86c27991a88978dea2fdf66397a`。

## 三类版本选择

在 408 条既有训练问句上加入 72 条新的否定与版本顺序对照。冻结模型和原入口的读入判断，只拟合三类版本分数。用 76 条校准问句分别选择语义版和语义加字符版的最低修正置信度。72 条旧留出正例的时间分类从 44 条升到 50 和 51 条。96 条另一组留出上分别从 58 条升到 85 和 82 条。

把语义版接到已重载的窗口检查点，96 条留出的端到端换值从 **47/96 升到 71/96**。逐题新增 24 条正确，原先答对的全部保留。32 条负例仍有 2 条触发读取，读入和人名结果逐题一致。改进集中在当前和改前两个类别。在长文本换问法的 12 条上，时间判断从 0 到 1，最终换值仍为 0。训练和端到端实验使用的都是真实模型隐藏状态，模型主权重未参与优化。完整记录在 `runs/spora-anchored-temporal-selector-v1/`、`runs/spora-anchored-temporal-selector-lexical-v1/` 和 `runs/spora-anchored-broad-temporal-v1/`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$runArgs = @('--checkpoint', 'runs/spora-native-query-key-v1/bootstrap.pt',
  '--route-source', 'runs/qwen-compatible-gelu-layer17-v1/integrated.pt',
  '--router', 'runs/qwen-compatible-gelu-lexical-boundary-v1/hybrid',
  '--temporal', 'runs/qwen-compatible-gelu-temporal-factorized-v1/factorized')
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_natural_memory.py @runArgs --output runs/spora-anchored-natural-memory-v3/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_natural_memory.py @runArgs --route-current-turn --output runs/spora-anchored-natural-memory-separated-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_natural_memory.py @runArgs --route-current-turn --output-gains 1.5 2 --output runs/spora-anchored-natural-memory-gains-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_natural_memory.py @runArgs --route-current-turn --heldout --output-gains 1 1.5 --output runs/spora-anchored-natural-memory-heldout-v1/result.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_natural_memory.py @runArgs --route-current-turn --output-gains 1.5 --pack-to runs/spora-anchored-native-candidate-v1/bootstrap.pt --output runs/spora-anchored-native-candidate-v1/packed.json
$runArgs[1] = 'runs/spora-anchored-native-candidate-v1/bootstrap.pt'
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_natural_memory.py @runArgs --route-current-turn --output-gains 1.5 --reload-only --output runs/spora-anchored-native-candidate-v1/reloaded.json
$runArgs[7] = 'runs/spora-past-override-v1'
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_natural_memory.py @runArgs --temporal-source runs/spora-native-query-key-v1/bootstrap.pt --route-current-turn --heldout --output-gains 1.5 --reload-only --output runs/spora-anchored-natural-memory-heldout-override-v1/result.json
$runArgs[7] = 'runs/qwen-compatible-gelu-temporal-factorized-v1/factorized'
.\.venv\Scripts\python.exe -u scripts/audit_spora_temporal_selector.py --checkpoint $runArgs[1] --temporal $runArgs[7] --temporal-source $runArgs[3] --output runs/spora-anchored-temporal-selector-v1
.\.venv\Scripts\python.exe -u scripts/audit_spora_temporal_selector.py --checkpoint $runArgs[1] --temporal $runArgs[7] --temporal-source $runArgs[3] --lexical --output runs/spora-anchored-temporal-selector-lexical-v1
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_natural_memory.py @runArgs --broad-versions --output-gains 1.5 --reload-only --output runs/spora-anchored-broad-temporal-v1/baseline.json
.\.venv\Scripts\python.exe -u scripts/audit_spora_anchored_natural_memory.py @runArgs --broad-versions --output-gains 1.5 --reload-only --version-selector runs/spora-anchored-temporal-selector-v1/selector.pt --output runs/spora-anchored-broad-temporal-v1/selector.json
```
