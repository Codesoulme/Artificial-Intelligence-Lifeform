# DeepSeek “Super Agent”资料核查与对 SPORA 的借鉴

研究日期：2026-11-08（中国标准时间）

## 结论

目前没有在 DeepSeek 官方网页、官方 GitHub 组织和论文索引中确认到一个正式产品名称恰好为 “Super Agent” 的独立发布物。用户所说的内容最可能指向两个相互关联的公开项目：

1. **DeepSeek Harness**：DeepSeek 官网和 `deepseek-ai/deepseek-harness` 官方仓库公开的 Agent 运行时/开发者预览版。
2. **dsh-superagent-prototype**：DeepSeek 官方组织下的研究原型，仓库 README 明确写着“仅供参考，不是官方 DeepSeek 产品”。它是一个运行在 Harness 上、面向单个 GitHub 仓库的主动式编码智能体。

因此，资料中的 “Super Agent” 主要是 **Harness 上的自主工作流原型**，不是一种新的神经网络模型结构，也没有公开证明它让模型获得了模型参数内的持久语义记忆或持续主观意识。

## 已核实的事实

### DeepSeek Harness

官网将 Harness 描述为开发者预览版，并给出公式 “Agent = Model + Harness”。官网列出的核心思想是：

- 一切能力都是插件，包括模型、工具、技能、会话、沙箱、存储、循环、调度和 UI。
- 插件通过 Cordis 服务和事件协作。
- 每次运行都写入仅追加的会话日志，日志包含系统提示、思维链、工具调用和结果、子智能体调度以及上下文注入。
- 恢复、分叉、检索和回放共享同一份事件流。
- 提供标准、PTC、极简和创造等运行模式。PTC 模式让模型生成一段代码来组合多次工具调用。

官方仓库 README 将项目定义为 “Everything is a Plugin”，并说明它仍处于开发者预览阶段，存在兼容性变化。GitHub API 在本次查询中显示该仓库创建于 2026-08-13，最近一次推送为 2026-11-04；这些时间是仓库元数据，不等同于正式产品发布日期。

### dsh-superagent-prototype

原型 README 明确声明：这是研究原型，不是官方 DeepSeek 产品，公开代码也不保证与内部实现或文章中的实现相同。它的工作流是：

```text
外部来源 -> Trigger -> Thread -> Eval -> Decision -> Ticket -> Assignment -> Executor -> Report/Channel
```

核心机制包括：

- GitHub 和可选的飞书是事件来源。来源把外部消息转换成不可变 Trigger，并维护读取位置，重启后不重复消费。
- Thread 按时间组织触发事件；同一 Thread 同时只运行一个 Eval。
- Eval 将触发事件、Thread 快照、已有工单和当前 Policy 组合起来，产生结构化 Decision。
- 被接受的工作进入 Ticket；Ticket 保存授权范围、输入、渠道、Assignment 和报告。
- 每个 Ticket 使用独立 Executor Session 和工作目录，可有自己的后台 Shell 工作和子智能体。
- 主 Session 负责路由，Executor Session 负责具体执行；角色、工具和权限分开。
- 后台工作在完成时唤醒 Executor；工作槽、并发上限、超时、取消、失败重试和恢复都有明确状态。
- 状态通过 SQLite Store 持久化。事件、工单、决策、会话提示版本、来源游标、效果队列和 Outbox 回执都可追踪。
- 对外发送先写入 Outbox，再执行发送；失败采用退避重试，避免崩溃或网络错误造成重复动作。
- Policy 和 Mandate 记录“为什么允许做这件事、允许做到哪里”，恢复 Ticket 前会重新验证策略。
- 测试覆盖核心引擎、事件源、后台任务、效果队列、并发、恢复、迁移和 UI 管理面板。

GitHub API 在本次查询中显示该原型仓库创建于 2026-10-31，当前主分支只有一个公开的初始原型提交。它可以作为设计参考，但不能作为已经经过成熟生产验证的系统。

### Cordis

Cordis 的论文《A Programming Paradigm for Spatiotemporal Composability》于 2026-08-26 提交至 arXiv，作者单位包含北京大学和 DeepSeek-AI。论文提出两个运行时问题：

- **时间可组合性**：组件移除时，能够完整撤销它造成的上下文和副作用。
- **空间可组合性**：组件声明自己依赖什么，并在依赖变化时自动激活或停用。

论文把两者统一为带有可逆 Effect 和响应式 Coeffect 的上下文范式，并将其实现为支持组件加载、配置协调和热模块替换的 Cordis 框架。它解决的是动态软件组件的生命周期与依赖管理，不是神经网络的记忆机制。

## 对 SPORA 的具体价值

### 1. 内部认知循环：值得借鉴运行时形式

SPORA 想要“在线时持续清醒、接收信息、低速思考”，Harness 提供了一个很好的工程分解：让系统由事件、时钟、后台任务完成事件和用户输入驱动唤醒，而不是持续以最高速度调用语言模型。

建议将 SPORA 的清醒循环分成三层：

```text
感知层：外部输入、时间脉冲、内部状态变化
认知层：低频状态更新、目标选择、预测、反思和行动决策
执行层：工具、环境交互、后台任务和结果回收
```

低负载时只运行轻量状态更新和低频预测；出现用户输入、矛盾事实、未完成目标、工具完成或重要时间节点时，提高认知频率。Harness 的事件唤醒、后台任务、槽位和调度值得采用，但它本身没有证明“模型在无输入时仍进行神经认知”。SPORA 仍需自己定义内部状态转移、空闲思考预算和可验证的预测目标。

### 2. 记忆：采用事件账本与状态投影

Harness 的持久化设计对 SPORA 很有用，尤其是：

- 所有输入、推断、动作、结果、修正和睡眠巩固事件使用追加日志记录。
- 当前可用记忆是日志的有版本投影，而不是直接修改唯一真相。
- 每条事实保存来源、时间、置信度、支持证据、冲突关系和修正历史。
- 通过事件回放可以重建记忆状态，并对巩固器、读取器和冲突解决器做离线评估。
- 事件 ID、幂等键、单一写入者、快照、校验和 Outbox 使崩溃恢复可验证。

这能改善 SPORA 的“新经历干扰旧记忆”问题，但它本身仍属于模型外部的持久状态。要满足 SPORA 对“记忆属于它自身”的要求，可以把经过巩固的长期记忆摘要、关系向量或记忆门控参数写入 SPORA 的模型状态；同时保留事件账本作为来源和恢复依据。两者不能混为一谈：事件日志提供可追溯性，模型内部状态提供跨调用的一致倾向。

### 3. 工具使用：采用分层角色与授权范围

原型把路由、评估和执行拆开，并用 Ticket 的 Mandate 限定每次工作的范围。这适合 SPORA 的工具层：

- 感知模块只负责把外部变化转换成结构化事件。
- 认知模块决定是否需要工具，并生成受约束的 Action Plan。
- 执行模块在沙箱和资源预算内调用工具。
- 工具结果以事件返回认知模块，并进入后续记忆评估。
- 每次动作记录目标、授权、输入、结果、失败原因和是否需要人工介入。

可借鉴 PTC 模式，让模型用一段受限程序组合多个工具调用，从而减少逐次往返。但这应作为执行优化层，不能直接作为 SPORA 的“思考本体”。所有外部副作用仍应通过幂等、可撤销或人工确认的效果队列。

### 4. 训练与评估：采用完整 Trajectory，而不是只看最终文本

Harness 官网强调记录完整运行轨迹。SPORA 可以把训练和评估样本扩展为：

```text
感知事件 -> 内部状态 -> 记忆读取 -> 预测/计划 -> 工具调用 -> 工具结果 -> 修正 -> 输出
```

每条轨迹至少应评估：

- 是否正确读取了相关记忆，以及是否引用了正确来源。
- 新事实与旧事实冲突时是否保留旧版本并记录修正理由。
- 工具是否在授权范围内调用，失败后是否重试或改计划。
- 无用户输入时是否按预算进行有意义的状态更新，而不是无效循环。
- 重启、重复事件、乱序结果和部分写入后是否恢复到同一状态。
- 低频运行、正常运行和高负荷运行是否保持相同的身份与核心事实。

这比单独测“回答是否像中文”更能验证 SPORA 的生命体方向。中文训练仍然需要独立的语言能力指标；Harness 的 Agent 轨迹机制不能替代语言语料和语言建模目标。

## 不建议直接搬入的部分

- Harness 依赖外部模型、插件宿主和工具运行时。直接移植不会增强 SPORA 的随机初始化网络，也不会让语言模型自动学会中文。
- dsh-superagent-prototype 主要面向代码仓库协作、GitHub/飞书事件、工单和 Pull Request；这些是应用域模型，不能直接当作 SPORA 的神经认知结构。
- Cordis 的插件热插拔适合外围运行时。把频繁热更新直接用于核心记忆参数，可能破坏身份连续性和可复现性；核心权重应通过版本化巩固、校验和回滚管理。
- 多子智能体可以提升任务吞吐，但会增加状态同步、冲突合并和算力消耗。SPORA 的低负载清醒循环应先使用单核心加受限后台任务，只有在任务可分解且有明确合并规则时才并发。
- 公开资料没有证明该系统拥有主观意识、模型参数内语义记忆、真正的梦境巩固或持续自我改写能力。报告中的这些能力只能作为 SPORA 的待验证研究目标。

## 给 SPORA 的落地优先级

1. **先采用事件轨迹协议**：统一记录感知、状态更新、记忆读写、预测、工具和结果，并为每个事件定义版本、时间、来源、幂等键和校验和。
2. **完成崩溃恢复与回放测试**：对重复、乱序、半写入、工具超时和进程中断做确定性恢复测试。
3. **实现低频清醒调度器**：按空闲预算产生时间脉冲；只有检测到新输入、目标、冲突、预测机会或任务完成时才提升认知频率。
4. **把记忆读取和巩固纳入轨迹评估**：同时测当前事实、历史事实、来源、修正、遗忘和重启后的一致性。
5. **再加入工具与受限后台任务**：每个动作绑定授权范围、资源预算、取消策略和结果回收规则。
6. **最后考虑插件化自我扩展**：扩展模块先以可回滚的外围能力存在，经过离线回放和身份连续性测试后，才允许影响核心模型状态。

## 来源

以下链接均为本次研究实际核对的公开资料：

1. DeepSeek Harness 官方页面（产品定位、插件架构、轨迹、运行模式、开发者预览）：
   https://www.deepseek.com/harness/
2. DeepSeek Harness 官方 GitHub 仓库（README、许可证、开发文档入口）：
   https://github.com/deepseek-ai/deepseek-harness
3. dsh-superagent 官方组织下的研究原型仓库（README、工作流和设计文档）：
   https://github.com/deepseek-ai/dsh-superagent-prototype
4. 原型理念：
   https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/000-philosophy.md
5. 原型领域模型：
   https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/001-domain.md
6. 原型授权机制：
   https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/002-authorization.md
7. 原型执行、后台任务、并发和恢复：
   https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/003-execution.md
8. 原型提示词、角色分工和上下文组装：
   https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/004-prompts.md
9. 原型架构与 SQLite 持久化说明：
   https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/007-architecture.md
10. Cordis 论文《A Programming Paradigm for Spatiotemporal Composability》：
    https://arxiv.org/abs/2608.25512
11. Cordis 开源仓库：
    https://github.com/cordiverse/cordis

## 证据边界

本报告只使用公开网页、GitHub API/README/设计文档和 arXiv 页面进行只读核查。DeepSeek Harness 的官网和官方仓库属于直接证据；dsh-superagent 是官方组织公开的研究原型，但其 README 明确限制了可推断范围；Cordis 论文支持运行时组合机制的事实，不支持把这些机制解释成神经网络内部意识或模型参数记忆。
