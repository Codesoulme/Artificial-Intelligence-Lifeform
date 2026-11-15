# 可学习自然写入器审计

## 实验一：有限值训练

写入器读取事实值所在位置的冻结底座隐藏状态，训练线性投影重建值 token 向量。训练使用 10 个饮品值，留出 4 个值；训练后写入器重建余弦均值约 `0.998`，留出值约 `0.062`，通过序列槽生成的两个留出事实完整正确数为 `0/2`。

这证明前一轮序列槽的成功来自直接保存底座 token 向量，而不是当前写入器从隐藏状态中学会了通用语义转换。

## 实验二：广覆盖 token 训练

为排除有限饮品过拟合，使用 256 个稳定中文单 token、2 名人物训练，另用 64 个 token、2 名人物留出。训练集写入余弦均值为 `0.992`，留出集只有 `0.099`，留出 token 没有一个达到 `0.9`。

因此，当前底座最终隐藏状态到词向量的简单线性写入器不能作为可靠的开放值写入方案。继续增加同类训练步数或饮品样本没有根据。

## 决策

当前主路线保留底座自身 token embedding 作为文本写入编码：它不维护固定值词表，值来自事实文本的 tokenizer 片段，并已经通过多 token、纠正、未知路由和独立重载审计。可学习隐藏写入器暂不接入正式槽状态。

相关产物：

- `spora/semantic_text_writer.py`
- `scripts/deepseek_spora_learned_natural_writer_pilot.py`
- `scripts/deepseek_spora_text_writer_broad_generalization.py`
- `runs/deepseek-spora-learned-natural-writer-20261116/result.json`
- `runs/deepseek-spora-text-writer-broad-generalization-20261116/result.json`

## 尚未解决

写入器没有形成在线自监督更新，事实解析句式仍有限，完整自由聊天尚未验收。下一步先交付当前可用的本地交流入口，再用真实多轮数据研究文本级写入目标，而不是继续调这个线性投影。
