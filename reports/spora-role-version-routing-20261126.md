# 从中文问句选择记忆角色与版次

上轮已验证，用共享读出能让答案不随两条记忆的输入顺序改变。那一轮由实验直接给出角色与版次。本轮在已确认记录对象、已知其两个候选角色的情况下，让孢子从末问决定每个角色是否读取，以及读取当前版还是首版。模型只看问句和候选角色名称，版本值在路由完成后才从快权重重建。新规划头使用冻结 DeepSeek 前十二层的中文表示。选出版本后，沿用孢子已有的时间轴查询和共享成句头。 [上一轮版本读出](spora-revision-roles-20261126.md) · [Lost with a Map](https://arxiv.org/abs/2609.33883) · [Xing USP](https://arxiv.org/abs/2610.34343)

训练使用 216 条自编请求，覆盖两项读不同版次、两项读同版次、只读一项和无需读取。测试使用另外 48 条请求，改变角色表达和版本措辞，其中 36 条需要读取，12 条不需要。训练和测试的主题相同。位置对照交换两个候选角色及其对应特征，再比较动作是否随角色移动。报告分别记录计划、实际数据库版本和回答里使用的事实。某些保持不变的字段只有一份记录，读「首版」和「当前版」可能得到同一个版本编号，因此版本编号正确数可能略高于计划正确数。 [StaleBench](https://arxiv.org/abs/2611.19752)

第一组将候选角色附在问句末尾，只用末端角色表示做三分类。第二组让该角色表示同时对问句正文做注意力，仍复用同一冻结模型、训练请求、初始化种子、220 步优化、版次逻辑和成句权重。第二组增加了查询和键投影。两组在本地实验版本库读取，路由仍需逐项复核。

| 指标 | 角色末端表示 | 加问句正文注意力 |
| --- | ---: | ---: |
| 两个候选角色的动作均正确 | 15/48 | 10/48 |
| 实际选择的版本编号全部正确 | 18/48 | 10/48 |
| 双项请求的完整动作正确 | 1/24 | 6/24 |
| 单项请求的完整动作正确 | 4/12 | 0/12 |
| 无需读档的请求误读 | 2/12 | 8/12 |
| 应答事实都出现、竞争值未出现且正常结束 | 8/36 | 6/36 |

两组都能让动作随候选角色换序而移动。抛开排列问题后，正文注意力多解出了五道双项请求，但对单项和无需查档的措辞泛化更差。两组都做过错误读取，候选保留在研究入口。两组具名记忆回归为 32/32，源文件和个体摘要一致，分别耗时约 107、120 秒。81 项版本、循环注意力和持久化相关用例通过。生成指标按事实文本匹配，未给成句自然度打分。

下一轮将沿用孢子已有的读取确认路径，把「是否要查」和「每项该读哪一版」分成两个决策，再单独测试明确否定、引用中的查档描述及同一句里对两个角色的不同时间要求。相关研究发现，仅从当前问句很难恢复更早确认的对象和版本约束。可另外用已确认的结构化会话状态提供缺省坐标，仍从内存版本张量读事实。 [Parametric Session States](https://arxiv.org/abs/2610.33047)

## 复现

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
$parent='runs/spora-composition-late-gates-20-23-24-v1/candidate.pt'
$reference='runs/spora-revision-version-state-v1/result.json'
$reader='runs/spora-revision-roles-shared-v2'
$layout='runs/spora-hybrid-position-recovery-v1/19-22'
$layers=@("$layout/layer-19", "$layout/layer-22")
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_role_version_routing $parent $reference $reader runs/spora-role-version-routing-v1 @layers
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_role_version_routing $parent $reference $reader runs/spora-role-version-routing-attention-v1 @layers --role-attention
```
