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

## 在复杂更正对话候选上迁移第 26 层

从 `runs/named-memory-dialogue-midlayers-composition-v1/integrated.pt` 起步，替换第 26 层的前馈门控。143 道 C-Eval 题逐项一致，得分均为 49/143。32 条受控记忆答案逐项一致，全部通过。24 轮开放开发对话都正常结束，22 轮生成词元一致。发生变化的是早餐题的首轮和第三轮，仍有未满足的食材要求。

写入独立孢子检查点后，逐项检查了已有张量、个体数据和结构配置。只有第 26 层门控清单变化。独立进程重载后，143 道题、24 轮对话和 32 条记忆答案复现。另一组 12 段短多轮保持 4/12 组、17/26 轮，12 段三轮更正的末轮保持 4/12，前文反转保持 5/8。

检查点和逐题结果位于 `runs/spora-composition-gelu-layer26-v1/`，候选摘要为 `d1845f5837cd0a780c61258ade0a77ec248f8bf8f7137a8566775db11e9d4d73`。这些开发对照用于确定逐层迁移的起点。

## 累积迁移第 21 和第 26 层

从只迁移第 26 层的检查点继续迁移第 21 层。两版 C-Eval 均为 49/143，两个原已答错的选项发生变化。受控记忆逐项一致，仍为 32/32。完整短多轮保持 4/12 组、17/26 轮，三轮更正末轮保持 4/12，前文反转保持 5/8。与无 GELU 的复杂更正候选相比，26 轮短对话文本和 24 轮开放开发输出均一致。两处迁移都在隐藏状态索引 6 和 18 的时间判断特征之后。

第二次封装只增加第 21 层门控记录。独立进程逐项复现 C-Eval、开放开发对话和受控记忆，检查点摘要为 `19b68738f41df5e979e07c7cd8fd9145041fbc13e6febb00ecc9b0fbca7dcb81`。逐题记录位于 `runs/spora-composition-gelu-layer26-layer21-v1/`。

本地入口可试用含上述两层门控的检查点。加载时检查已有张量、原始记忆绑定及门控位置，仍以原封装检查点标识个体。这个开关的验收范围包括两个记忆读取位置、两种时间判断特征及已经测过的读取方式。

用两个本地入口进程先后打开同一实验库副本。启用双层迁移的进程读回当前和旧饮品、当前和旧集合点，拒绝未登记近名，五条回复与原入口逐字一致。比对摘要确认只读复查前后副本一致，来源数据库也一致。额外的三轮对话按要求给出改期提醒和地点。复查脚本初次把加了「请」的提醒当成错误，接纳该等义写法后重跑通过。项目短多轮表仍用冻结的严格规则。相关单元测试 39 项通过。

```powershell
Set-Location D:\SPORA
.\.venv\Scripts\python.exe -m scripts.spora_named_memory_chat --dialogue-checkpoint runs/spora-composition-gelu-layer26-layer21-v1/candidate.pt
```

## 后层门控的逐层筛选

以第 21、26 层检查点为共享对照，依次试验第 20、23、24 层。保留已通过的替换，回滚不通过的替换。筛选要求受控记忆输出逐项一致，且 C-Eval 总分、对话结束数、严格组通过数和逐轮通过数均不低于共享对照。每层使用相同的原题和贪心解码设置。

| 新增门控 | C-Eval | 开发对话词元与对照一致 | 正常结束 | 严格组 / 轮 | 受控记忆 |
|---|---:|---:|---:|---:|---:|
| 双层对照 | 49/143 | 24/24 | 24/24 | 4/12 · 17/26 | 32/32 |
| 第 20 层 | 48/143 | 21/24 | 24/24 | 4/12 · 17/26 | 32/32 |
| 第 23 层 | 49/143 | 21/24 | 23/24 | 4/12 · 17/26 | 32/32 |
| 第 24 层 | 49/143 | 24/24 | 24/24 | 4/12 · 17/26 | 32/32 |

纳入第 24 层，三个门控位置为 21、24、26。第 24 层答错的一个选择题换了选项，受控记忆的实际输出逐项一致。临时替换第 23 层时，早餐建议用完 96 词元仍未正常结束。第 20 层和第 23 层各自恢复 SiLU。实验目录为 `runs/spora-composition-late-gates-20-23-24-v1/`。

独立重载核对了检查点的全部张量及门控清单，复现了选择题选项、开放对话词元、严格短对话和受控记忆。新检查点摘要为 `99fe05df3af8d6fa99b248d3e025016b62bfb351b014fa7030745b2494a0d8b2`。

另写了 8 组三轮日常开发题，按场景要求逐条阅读回复。三层版与双层版有 19/24 轮文本相同，两版均为 23/24 轮正常结束、0/8 组完成全部要求。三层版在解释车窗雾气时从「蒸发」改为「凝结」，最后仍漏掉冷玻璃。在安慰朋友的场景中，首轮给出了安慰，后两轮原样复述，未落实口语和字数要求。两版都在发通知时把推迟后的十点半答成十点。新题作为开发诊断，后续验收需要更广的独立场景及人工复核。

```powershell
$sporaParent = 'runs/spora-composition-gelu-layer26-layer21-v1/candidate.pt'
$sporaRun = 'runs/spora-composition-late-gates-20-23-24-v1'
.\.venv\Scripts\python.exe -m scripts.screen_spora_gelu_layers --checkpoint $sporaParent --ceval runs --output $sporaRun --layers 20 23 24
.\.venv\Scripts\python.exe -m scripts.screen_spora_gelu_layers --checkpoint $sporaParent --ceval runs --output $sporaRun --verify
.\.venv\Scripts\python.exe -m scripts.audit_named_dialogue_composition $sporaParent "$sporaRun/open-parent.json" --development-only data/conversation-development-v3.json
.\.venv\Scripts\python.exe -m scripts.audit_named_dialogue_composition "$sporaRun/candidate.pt" "$sporaRun/open-candidate.json" --development-only data/conversation-development-v3.json
```
