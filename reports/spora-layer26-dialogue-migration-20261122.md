# 第 26 层门控迁移与中文多轮对照

以 `runs/qwen-compatible-gelu-packed-reply-v1/packed.pt` 为起点。该检查点已将两处神经记忆读出迁入前馈输出，并持久化第 17、21 层的 GELU 门控和整词记忆回答。新实验仅将第 26 层的 SiLU 换成同样的缩放 GELU，层号从零起算。

## 评测方法

- 用相同的贪心解码和 96 词元上限，比较迁移前后在八个三轮场景中的回答。检查回答是否完成任务、是否接住第二和第三轮、是否遵守显式格式。场景来自既有开发集，曾用于观察候选表现。
- 同时比较两组 C-Eval 题的选项，分别记录 60 道语文、逻辑、数学题和 83 道额外学科题。
- 用两个留出人物、每人两种问法、每种问法八个值，检查变更内部状态时答案是否随之改变。评测提供已有地址和查询版本，测试的是记忆读出和作答。
- 原始 DeepSeek 使用相同的八个场景，分别测关闭思考段的贪心生成，以及开放思考段、温度 0.6、top-p 0.95、最多 256 词元的采样生成。开放条件只以思考结束后的文字作为可见回答。

## 第 26 层结果

| 配置 | 三科 C-Eval | 额外四科 C-Eval | 自然结束 | 记忆短答 | 四个问句都能随八份状态逐一换值 |
|---|---:|---:|---:|---:|---:|
| 已迁移第 17、21 层 | 24/60 | 23/83 | 18/24 | 32/32 | 4/4 |
| 再迁移第 26 层 | 24/60 | 23/83 | 18/24 | 32/32 | 4/4 |

两组选择题的 143 道题逐题一致，受控记忆也逐项一致。24 轮对话中有 18 轮逐词元一致，其余六轮换了说法。在变化的轮次里，早餐依然含鸡蛋，七张纸取走两张依然没有得到五张，雨天计划依然没有按用户给出的条件改成在家看电影。两套候选均未完整完成这八个场景中的任一场景。该单次配对共用时约 343 秒，CUDA 峰值分配 4.94 GiB。

把第三处 GELU 写入独立的 `candidate.pt` 后，新进程复现了临时迁移时的 143 道题选项、24 轮生成词元和 32 条记忆答案。检查点与起点相比只有门控层清单新增了第 26 层，全部张量逐元素相同。该候选用于接续结构实验。正式聊天入口仍需通过新场景验收。检查点 SHA-256 为 `bbc0231bf426405c35d3fc2649c7ff9c88436178b4ed1acda1cbe07b55e06bd7`。

## 原始底座和解码方式

| 原始 DeepSeek 的条件 | 生成可见回答 | 以结束词元收尾 | 完成三轮任务 |
|---|---:|---:|---:|
| 关闭思考段，贪心，最多 96 词元 | 24/24 | 18/24 | 0/8 |
| 开放思考段，采样，最多 256 词元 | 24/24 | 23/24 | 0/8 |

两种条件都会在无蛋早餐建议中加入鸡蛋，也都会把改约请求答成联网建议。开放条件在雨天追问中提到了留家看电影，但仍无法完成整个场景。实验同时改变了模板、采样和输出长度，因此这组数字只比较两套生成配置。开放条件采用一个固定种子，256 词元上限也限制了结果的适用范围。原始模型在关闭思考段的配置下，与迁移前检查点在 24 轮中有 15 轮逐词元相同，与第 26 层候选有 14 轮相同。

此前保存的本机 Qwen2.5-0.5B-Instruct 对照使用同一组题和 96 词元上限。它正确承接了邮局与菜市场的三轮顺序变更，也答出取走两张后还剩五张。它在人物借还、改约和雨天计划上仍有实质错误。两份底座各用自己的对话模板与模型设置。 [同场景 Qwen 记录](qwen-instruct-holdout-20261121.md)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe scripts/audit_spora_dialogue_migration.py migration --checkpoint runs/qwen-compatible-gelu-packed-reply-v1/packed.pt --ceval runs --output runs/spora-language-migration-layer26-v1/migration.json
.\.venv\Scripts\python.exe scripts/audit_spora_dialogue_migration.py raw --checkpoint runs/qwen-compatible-gelu-packed-reply-v1/packed.pt --output runs/spora-language-migration-layer26-v1/raw.json
.\.venv\Scripts\python.exe scripts/audit_qwen_gelu_migration_20261121.py --checkpoint runs/qwen-compatible-gelu-packed-reply-v1/packed.pt --ceval runs/ceval-validation-617524a --layer 26 --candidate runs/spora-language-migration-layer26-v1/candidate.pt --output runs/spora-language-migration-layer26-v1/packed.json
.\.venv\Scripts\python.exe scripts/audit_spora_dialogue_migration.py verify --checkpoint runs/spora-language-migration-layer26-v1/candidate.pt --parent runs/qwen-compatible-gelu-packed-reply-v1/packed.pt --migration-report runs/spora-language-migration-layer26-v1/migration.json --ceval runs --output runs/spora-language-migration-layer26-v1/reload.json
```

## 对更大结构改动的参考

MOHAWK 将迁移分为词元混合、块输出和整网预测三个目标。Llamba 用这一方法将训练好的 Transformer 迁成了不同的循环结构。孢子如果继续更换注意力计算，可以先用旧层记录的跨位置混合作为监督，再对齐输出残差和整网分布。三阶段各自独立验收，尤其要保留新问法和当前／历史换值的对照。两篇论文的结果来自有规模的跨架构训练，本地单卡实验先验证一个部件。 [MOHAWK](https://arxiv.org/abs/2408.10189) · [Llamba](https://arxiv.org/abs/2502.14458)
