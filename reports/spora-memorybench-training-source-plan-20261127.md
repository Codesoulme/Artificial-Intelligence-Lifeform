# MemoryBench 中文候选来源检查计划

2026-11-27。[MemoryBench 论文](https://arxiv.org/abs/2510.17281)研究从模拟用户反馈中持续学习；[固定数据卡](https://huggingface.co/datasets/THUIR/MemoryBench/blob/3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d/README.md)声明 MIT、英中混合，问答和反馈由模型模拟。数据卡列出多个源数据集；汇总数据卡的许可不自动确认每个原始语料及模型输出的全部权利。其目标不是人物记忆寻址，不能把反馈分数直接视作姓名、读写意图或事实真实性标签。

固定数据集版本 `3acd60a4bd35b43b408f0e6db4c5f1e88df5e96d`，只下载 JuDGE 和 WritingBench-Creative&Design 的训练分区，按服务端 LFS SHA-256 与字节数验证；不读取测试分区、不训练模型。用 Arrow 读取并统计语言、字段、轮次、模拟反馈结构、空值和来源标签；不将原始文本或训练数据上传。对字符串对象只使用 JSON 或 `ast.literal_eval`，不执行数据内容。输出是来源摸底，不宣称完成标注质量、权利链、真实用户或记忆意图验证。失败时保留错误信息和部分文件，不伪称下载成功。
