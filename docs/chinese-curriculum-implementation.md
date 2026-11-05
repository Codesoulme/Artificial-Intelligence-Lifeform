# P18：新课程继承、覆盖抽样与恢复

本轮已经按固定配置完成1000次新增更新。结果与边界见[P18报告](../reports/P18-report.md)。所有学习模块仍来源于随机初始化的本项目谱系，没有导入预训练语言权重。

## 数据与采样

配置位于 `configs/chinese-curriculum-1000.json`，源材料为 `data/chinese-pilot/basic-curriculum-v2.json`、原基础文本与用户要求的重点课程。`scripts/prepare_chinese_curriculum.py` 只产生数据，不运行优化器；组内含单词、教学短句与模板组合，结构先验在结果中明确披露。

`CoveringSampler` 给每个训练片段分配唯一位置，并将它归入 `focus` 或 `basic`。固定顺序为重点、基础、基础、重点、基础；每组使用自身的打乱队列，全部取完才重排。批次边界不会重置队列，因此跨批次也不会漏掉尚未访问的条目。它保证组内一轮覆盖，不保证每个单词的目标频率完全相同。

状态包含课程签名、随机状态、各组排列、遍历位置、总抽样次数和每片段计数。恢复时检查分组签名、完整排列、位置范围和计数一致性。实验结束独立复算同一顺序与目标数量，核对保存结果。

## 参数与谱系

`CurriculumTrainer.begin` 只允许从明确绑定的已训练接口开始一次。它复制先前权重，记录源文件摘要、先前步数及累计目标量，为改变后的课程建立新的优化器状态。本轮1000步独立计数，谱系总步数为1100，不能用新计数掩盖历史训练量。

`CurriculumTrainer.restore` 恢复同课程的权重、优化器、各类随机状态和遍历状态，并核对抽样次数与优化步数。普通训练器禁止接收这种检查点，以防缺失分组状态。新恢复路径已通过CPU连续／中断对照，GPU仅核实独立进程恢复与评估。

实际更新仍调用公共 `ByteTrainer.step_batch`，损失、梯度裁剪与AdamW保持原定义。这个提取没有改变旧随机抽样入口的行为，历史测试继续通过。父个体的全部参数均冻结；每100步对全部旧张量重新计算摘要。

## 执行与检查

以下是本轮已执行的命令，原目录存在时拒绝重跑，避免覆盖证据：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_chinese_curriculum.py
.\.venv\Scripts\python.exe scripts/train_chinese_curriculum.py --start-language-training
.\.venv\Scripts\python.exe scripts/evaluate_chinese_curriculum.py
.\.venv\Scripts\python.exe scripts/report_chinese_curriculum.py
```

训练脚本是这一份固定预算的实验入口；不提供自动扩大课程或无限继续的操作。检查点保存于 `runs/chinese-curriculum-1000/`，每10步原子更新主检查点，每100步另存一份。结果中断时，库层可以恢复已保存的课程状态；若要制作新的恢复执行脚本，需保留原输出并扣除已执行步数，不能重新获得1000步预算。

每100步使用八篇开发文本、重点课程损失及六条预先固定提示。开发损失超过本阶段基线20%或旧状态变化时停止；第1000步固定结束。新封存文本只在最终评估读取，不选最优中间版本。旧18题已公开，作为持续诊断；额外6题在本轮更新前固定，所考事实依然来自同一有限课程。

原始字节生成不做UTF-8限制或修复。另行统计合法率与内容正确率，记录只有标点、短输出及相同答案现象。不能把合法编码的增加当成中文理解已经成立。
