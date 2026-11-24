# 孢子解码器的字段许可与缺资料动作

## 改动

在原整字段解码之前，增加用问题预测字段角色的小模型。每个槽位可以是必需、可选、明确排除或未提及。只把必需和可选的字段交给读取器，缺少可选字段不会触发缺资料答复。观察相关性和读取意图沿用上一轮已训练的入口。 [上轮实验](spora-observation-read-gate-20261124.md)

缺少必需字段时启用另一组动作分数。模型可以生成普通词、复制可用字段、输出指定字段的「尚未记录」标记或结束。标记有固定的短语，选择哪个标记及何时说由模型决定。标记和被复制的值都作为真实词元写入自回归状态。完整状态继续使用原整字段动作头。普通文本生成仍可自行写出词表中出现的内容，因此仍要检验答复本身有无编造。

字段角色监督来自孢子问题、同主题的排除与可选成对句。外部数据仅为地址是否被请求提供辅助标签。用固定提交的 CrossWOZ 训练文件前 1 MiB，解出 384 段完整训练记录，分领域和标签抽取 96 条地址监督，未把其他字段的地址标记当作它们的训练标签。缺资料动作另用孢子训练值组成单项缺失和全缺样本。两次训练都冻结原语言骨干、原动作头和个体。 [CrossWOZ](https://aclanthology.org/2020.tacl-1.19/) · [固定数据版本](https://github.com/thu-coai/CrossWOZ/tree/df82c9fdff91b9b130f2d6b89110d3870ba6260e/data/crosswoz)

真实生成在原两批各 12 道换值题、6 道明确排除题、6 道可选题和 6 道带排除措辞的普通题上核对。状态的槽位值仍由试验提供。普通题对照使用同一语言头的空状态输出。

## 结果

字段角色用 288 条孢子问题和 96 条地址辅助样本训练 200 次。缺资料头用 108 条答复记录训练 384 次。下表的完整字段计划要求所有必需字段及其他字段的角色一并正确。双向通过要求同题的原值和换值答复都满足字段检查与重复检查。

| 题组 | 字段计划全对 | 上一轮双向通过 | 本轮双向通过 | 缺一字段时正确报告缺失且保留可用值 |
| --- | ---: | ---: | ---: | ---: |
| 此前只给问题的 12 题 | 10/12 | 6/12 | 5/12 | 2/24 |
| 新措辞、新值 | 4/12 | 4/12 | 0/12 | 0/24 |

明确排除题的完整角色正确率为 1/6。3/6 答复包含所请求字段，2/6 泄漏了被排除的值，这两题的读取计划均将排除项预测成必需项。可选题的完整角色正确率为 0/6，多数被当作必需，仅有 3/6 完整状态答复写出了必需值。6 道带排除措辞的普通题有 5 道复现空状态。上一轮的乐团招募误读恢复，另一次通用就医提醒误用了地点。效果没有达到替换原路径的标准，权重留在隔离试验中。

32 条持久记忆读取和此前逐条相同，个体摘要一致。31 项相关测试通过。训练与真实生成约 227 秒。逐题结果存于运行目录 spora-field-access-v1。

独立进程重载复现了普通题、字段角色、两组取值生成及持久读取。额外固定新题的正确字段计划再测缺资料动作。完全无记录的 6 题都报出了两项缺失并正常结束。只缺一项的 24 题也都报告了正确缺项，但没有一条读出已登记的字段，随后直接结束。完整答复的 0/24 在固定计划下仍然出现，问题同时涉及字段选择与生成。

## 结束顺序对照

固定正确字段计划，从同一权重分别增加两种约束。一种在必需字段仍未读取或报告时暂不允许结束；另一种还把可读必需字段放在缺项标记前。两种都从原有解码器选取下一步动作，不更新任何权重。

| 给定正确字段计划的新题缺一项 | 报告正确缺项、写出已登记值且正常结束 |
| --- | ---: |
| 原解码 | 0/24 |
| 暂停结束 | 24/24 |
| 暂停结束且先读可用字段 | 24/24 |

字段值和缺项检查对措辞不敏感。只暂停结束时，人工复核仍看到四条把人员姓名写成地点，以及两条在正确值之后重复或复述地点。先读可用字段也出现「数量为人名」和地点与钟点错配。由此可以确认，阻止过早结束能恢复动作覆盖，读出的语义角色还需要传入答复措辞。实验中的曲名仍沿用原型的物品槽，所以缺项用语也有本身的类别限制。

将只暂停结束的模式接回模型预测的字段计划，旧题缺一项对照的字段规则通过从 2/24 增至 18/24，新措辞从 0/24 增至 4/24。完整状态的实验成绩和排除字段失败仍按主表记录。后续应优先提高字段范围识别，并让正常生成获得动作的语义角色，以便把读出的正确值放在正确的句子位置。三份重载与对照记录都在运行目录。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$sporaZip = 'runs/crosswoz-train-prefix-df82c9.bin'
$sporaData = 'runs/crosswoz-address-prefix-v1.json'
$sporaUrl = 'https://raw.githubusercontent.com/thu-coai/CrossWOZ/df82c9fdff91b9b130f2d6b89110d3870ba6260e/data/crosswoz/train.json.zip'
Invoke-WebRequest -Uri $sporaUrl -UseBasicParsing -Headers @{Range='bytes=0-1048575'} -OutFile $sporaZip
.\.venv\Scripts\python.exe -u -m scripts.prepare_spora_crosswoz_requests $sporaData --prefix $sporaZip
$sporaBase = 'runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$sporaSplit = 'runs/spora-recurrent-split_gated_block_delta-warm-v1'
$sporaAction = 'runs/spora-span-actions-v1'
$sporaGate = 'runs/spora-observation-read-gate-v1'
$sporaOutput = 'runs/spora-field-access-v1'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_field_access $sporaBase $sporaAction $sporaGate $sporaData $sporaOutput "$sporaSplit/layer-19" "$sporaSplit/layer-22"
.\.venv\Scripts\python.exe -u -m scripts.train_spora_field_access $sporaBase $sporaAction $sporaGate $sporaData $sporaOutput "$sporaSplit/layer-19" "$sporaSplit/layer-22" --verify
.\.venv\Scripts\python.exe -u -m scripts.train_spora_field_access $sporaBase $sporaAction $sporaGate $sporaData $sporaOutput "$sporaSplit/layer-19" "$sporaSplit/layer-22" --audit-coverage
.\.venv\Scripts\python.exe -u -m scripts.train_spora_field_access $sporaBase $sporaAction $sporaGate $sporaData $sporaOutput "$sporaSplit/layer-19" "$sporaSplit/layer-22" --learned-coverage
```
