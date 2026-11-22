# 联合 Span 写入器研究

日期：2026-11-22

## 目的

当前 writer 分别预测起点和终点，容易把“周日”切成“日”。本轮原型把每个字段改成“无写入”或一个完整 `(起点, 终点)` span 的联合分类，并在 12 条读取计划留出题上测试自动状态。

## 结果

| 候选 | 自动状态完整正确 | 可渲染 | 原生精确回答 |
|---|---:|---:|---:|
| 当前 v1 writer | 8/12 | 11/12 | 9/12 |
| 联合 span，从随机权重 | 4/12 | 8/12 | 7/12 |
| 联合 span，继承 v1 表示和边界偏置 | 4/12 | 10/12 | 5/12 |

正确状态下两种联合候选的原生精确回答都是 `12/12`，说明失败集中在自动写入，不在计划渲染或原生解码路径。

对随机联合候选扫描 `margin=0、0.5、1、1.5、2`，自动状态均为 `4/12`；因此主要问题不是写入阈值。联合候选在训练损失下降后仍大量漏写地点、件数和保留字段，说明当前联合分类头没有得到足够的跨句 span 泛化。

## 决定

两种联合 span 候选均不接入正式检查点，继续使用当前 v1 writer。研究代码保留在 `spora/research_joint_span_writer.py`、`scripts/audit_spora_joint_span_writer.py` 和 `scripts/audit_spora_joint_span_margin.py`，用于后续比较。

下一轮不再简单扩大联合分类头；应改为在现有 writer 的有效边界上增加成对边界一致性损失，并单独加入“更新覆盖旧值”和中文复合值的留出监督。

结果文件：

- `runs/spora-joint-span-writer-v1/result.json`
- `runs/spora-joint-span-writer-v3/result.json`
- `runs/spora-joint-span-margin-v1/result.json`
