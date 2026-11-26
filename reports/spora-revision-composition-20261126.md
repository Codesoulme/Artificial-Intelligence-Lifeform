# 版本张量读回后的中文成句训练

上轮给定正确地址后，长对话中当前版和旧版都能完整读出，但答复多为两个短片段。本轮冻结第 19、22 层循环恢复版和个体状态，只训练输出动作头。输入来自同一份重开的版本库。参考研究将状态可读性、状态选择和最终行为分开测试，孢子继续把值的完整读取与写成用户可用的答复分别计分。 [上轮版本交换](spora-revision-version-state-20261126.md) · [Lost with a Map](https://arxiv.org/abs/2609.33883) · [CopyPasteLLM](https://arxiv.org/abs/2510.00508)

训练为 36 条自编答复，每条标注普通词元与完整字段动作，两个训练组使用同一初始化和 240 次更新。短问组仅看末问，混合组把 18 条扩展为约 900 词元，加入既有中文训练语料中的无关消息。两组共享问句、事实、答复和优化器，新增前文没有给出该样本的目标事实。测试复用八道约 1,400 词元的长问。每道固定整个输入，分别读当前、旧版和空状态。训练与留出问句和完整值不重合，话题类别相同。

## 长前文与受控换值

| 读出方式 | 当前版两值完整 | 旧版两值完整 | 当前版角色正确 | 旧版角色正确 | 两版角色均正确 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 上轮短片段头 | 8/8 | 8/8 | 0/8 | 0/8 | 0/8 |
| 短问成句训练 | 8/8 | 8/8 | 6/8 | 6/8 | 6/8 |
| 混合长度成句训练 | 8/8 | 8/8 | 6/8 | 6/8 | 6/8 |

严格规则检查两项各出现一次、旧值未混入、句子正常结束，以及具体值与角色动词正确配合。将最初自动统计的「有两个值和两个相关动词」改为按值的位置匹配后，从已落盘答复重新计算，两个训练组均为 6/8。这样才能识别「请把藤荫戏台放在周照杉收回」这类错配。市集的两种取值均失败，另三类未见取值能生成类似「请把二十一份样张交给薛临泉。」的完整句。两种训练在该小样本上打平。约 240 秒完成训练、权重重读和评测，32 条具名读回均通过，空状态与原普通答复逐条一致。

## 明确地址及版次的路由

将四个实验场景的内部地址映射到八个可明示的实验姓名。原版次路由直接读取消息中的姓名、字段和版本表达，返回数据库中已有的版本编号。当前与首版配对、按日读取与当前配对这两类请求各 8/8 均返回准确的两个版本。八条改写请求未读版本，八条含相近未知姓名的请求仅选择了另一条明确记录。可见姓名是实验显示名，字段和两个版次均明确写出。

在同一研究入口将实际返回的版本编号读回成张量词元，再交给短问组保存的权重。先同时输入查档条件和写作要求，当前版和按日旧版各 8/8 保住两个正确值，严格角色通过为 2/8 和 3/8。接着只给成句头写作要求，路由仍读取同一版本，两个版次的值仍各正确 8/8，严格角色通过均为 2/8。只移除查档措辞未改善结果。该入口把查档和写作要求作为两个显式输入段，未测试自由表述中自动分段或隐含地址的情况。

只给写作要求的条件加回固定长前文后，两个版次的严格角色通过均恢复为 6/8。16 条答复逐词复现了直接给定地址的成句评测。三个实验的路由、权重、版本库和所选记录一致。这组对照表明，保留会话能帮助按角色使用正确的值。下一轮应检查读出头能否直接利用已选记录的字段类型，在不给旧会话文字时区分「摆放地点」和「负责回收的人」，并加入同一类型对应两个事项的留出检验。 [状态可读性与实际行为](https://arxiv.org/abs/2609.33883)

79 项版本存储、时间、地址范围、循环注意力及聊天持久化测试通过。首轮落盘结果保留了初始宽松量表，报告里的严格分数使用最终脚本的角色判分复算。路由到成句的三份输出使用严格判分。

## 复现

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$parent='runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$reference='runs/spora-revision-version-state-v1/result.json'
$layout='runs/spora-hybrid-position-recovery-v1/19-22'
$layers=@("$layout/layer-19", "$layout/layer-22")
$output='runs/spora-revision-composition-v1'
.\.venv\Scripts\python.exe -u -m scripts.train_spora_revision_composition $parent $reference $output @layers
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_composition_routes $parent $reference "$output/routes.json"
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_composition_routes $parent $reference "$output/routed-generation.json" --composer $output --layers @layers
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_composition_routes $parent $reference "$output/routed-writing-only.json" --composer $output --layers @layers --writing-only
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_revision_composition_routes $parent $reference "$output/routed-writing-history.json" --composer $output --layers @layers --writing-only --with-history
```
