# RiSAWOZ 的中文状态变化与共指核对

本轮从发布方的固定版本取得原始对话、共指标注和分区清单。仅打开官方开发分区的成员，训练与测试的对话正文未读。原发布方明确将数据列为 CC BY-NC 4.0，GEM 镜像的数据卡声明 CC BY 4.0。本次采用原发布方数据的许可说明。对话正文与原始档案保存在本地运行目录，提交只包含统计、方法和脚本。 [发布方固定版本](https://github.com/terryqj0107/RiSAWOZ/tree/1e2d3be2b25a0540519db1664a172d85ddd62b29) · [GEM 数据卡](https://huggingface.co/datasets/GEM/RiSAWOZ/tree/d9287b21928a811281a349655655ee4be964292a) · [论文](https://aclanthology.org/2020.emnlp-main.67/)

逐轮比较相邻状态中同一领域字段的非空值。要求当前值同时出现在逐轮 `turn_inform` 及对应用户 `Inform` 行为中，再检查去除空白后的用户原话是否有唯一的新值字面跨度。按完整会话去空白后的字符序列与共指文本一一对齐，排除缺少匹配或匹配不唯一的样本。共指计数要求当前用户句有一个完整标注区间，且同一标注簇在更早回合已经出现。

| 开发分区统计 | 数量 |
| --- | ---: |
| 对话、用户回合 | 600、4,058 |
| 相邻状态中发生不同值覆盖的字段 | 153 |
| 与两份当前轮标注一致的变化 | 139 |
| 标注一致且原话含唯一新值字面区间 | 118 |
| 118 个字段事件覆盖的回合、对话 | 112、105 |
| 对话全文与共指档唯一匹配 | 530/600 |
| 具有唯一新值、且对话与共指档对齐 | 92/118 |
| 原话还包含指向先前回合的共指 | 15/92 |

检查了 15 个共指候选的原字段来源句和当前句。样本中有追加查询另一时间或地点、询问同一航班的另一舱位价格，以及提出假设行程的表达。即使这些样本在对话状态中覆盖旧值，共指标注只表明当前句引用了先前内容，无法独立确定用户授权把原安排永久改掉。这里的 118 个字段事件是可人工筛查的跨度候选，15 个带共指的事件可用于构造相近措辞的误写题，当前均未作为孢子的写入正例或训练数据。

两次运行输出文件逐字一致，SHA-256 为 `69b86f457db437accf7834d4a6c70605b458246880885c855e1f515ea7bf6caf9`。记录只包含定位所需的对话键、轮次、字段及共指判据。输出没有用户原话或标注值。

## 复现

从固定版本的 `RiSAWOZ-data` 目录取得 `source-data.zip`、`task4-data-Coreference resolution.zip`、`dataset_split_ids_dict.json`，保存到本地研究目录。脚本核对三份文件摘要和官方分区键集合。 [数据目录](https://github.com/terryqj0107/RiSAWOZ/tree/1e2d3be2b25a0540519db1664a172d85ddd62b29/RiSAWOZ-data)

```powershell
$risa = 'runs/research/risawoz-1e2d3be'
.\.venv\Scripts\python.exe -m scripts.audit_spora_risawoz_updates `
  "$risa/source-data.zip" "$risa/task4-data-Coreference resolution.zip" `
  "$risa/dataset_split_ids_dict.json" "$risa/update-audit.json"
```
