# 公开中文请求和新人名下的记忆查阅

上一轮将原句表示取平均，六条口语题中自主读对两条，校准后在另一组题上没有完成查阅。本轮对比扩大人工监督和引入公开中文请求两条路径。Xing USP 将动作、字段、实例归属和完整结果分别评估。ToolTurns 使用先追问再补充的序列评估整个交互。本地沿用分项评分，并加入含姓名近似、改口及版次的固定两轮回放。 [上一轮实验](spora-spoken-query-plan-20261126.md) · [Xing USP](https://arxiv.org/abs/2610.34343) · [ToolTurns](https://arxiv.org/abs/2611.14925)

## 对照设计

冻结当前两层循环版，复用第 12 层原问句平均特征和原版次读出头。两组使用同一批扩充的自编例子，加入新的问法、改口后绑定到第二个人、改用第一人的记录、以及同句编辑和撤回对照。双方再各加入 128 条「继续普通对话」监督。一组重复采样已有自编负例，另一组使用 SMP2020 和 CAIS 的公开原句。两组的模型、标签数量、随机种子和优化步数一致。训练和测试的公开原句按固定版本和分区抽取，使用类别筛选、去标点规范化、去重和确定性排序，训练和测试各取 128 条，逐条记录来源编号。

公开语料的原始标签用于抽样，例如导航、翻译、天气、点播和查询。在孢子的实验中，用户没有授权这些请求去查个人档案。这些样本用来检查误把普通请求当成查档或要求澄清。多人和版次查阅的正例以及带相似姓名的反例由本轮单独编写。

通过 Windows 入口在新档案登记十条短事实并退出，再用独立进程加载。新留出集含八条完整的口语查阅、六条无需查阅的相似措辞、两条需澄清的请求和三组补答或改口，每组两轮。第二轮输入包含用户的上一句和补充句，其中没有存入的具体值。整条回放要求第一轮选择追问且第二轮实际读对。该环节使用固定文本回放，追问的具体措辞和实时上下文变更留待聊天入口检验。新留出的人名及完整句式与训练和校准分开。

## 实测结果

两组新模型各使用 413 条共享监督及 128 条对照负例，校准使用 153 条。旧头使用上次保存的权重及阈值。

| 指标 | 旧头 | 新自编对照 | 新公开负例 |
| --- | ---: | ---: | ---: |
| 公开测试请求判为普通对话，阈值前 | 29/128 | 32/128 | 127/128 |
| 新口语题人物正确，阈值前 | 7/8 | 7/8 | 7/8 |
| 新口语题事项正确，阈值前 | 7/8 | 8/8 | 8/8 |
| 新口语题版次语义正确，阈值前 | 3/8 | 4/8 | 4/8 |
| 新口语题三项坐标同时正确，阈值前 | 2/8 | 3/8 | 3/8 |
| 新口语题实际读对，阈值前 | 2/8 | 3/8 | 3/8 |
| 22 条回放中的实际错读，阈值前 | 2 | 5 | 3 |
| 新口语题实际读对，阈值后 | 0/8 | 0/8 | 1/8 |
| 22 条回放中的实际错读，阈值后 | 0 | 0 | 0 |
| 完整补答或改口回放，阈值后 | 0/3 | 0/3 | 0/3 |

公开负例组在新口语题校准后唯一读对的值是「松风亭」。原句说「最初」，头预测「上一版」，此人恰好只有两版，所以读回同一条记录。把版次语义与记录一起要求正确时，此组校准后为 0/8。未加阈值时，三次错读中有两次涉及改口后的姓名或书目的版次。实际读档的名字、编号和生成值逐条保存在结果中。公开测试请求未绑定私人档案，127/128 衡量的是原始动作判断。它没有反映同名干扰时的实际读档准确率。

两个新头各在校准集选对了三次读取。它们在三组两轮测试中都能选择先追问，收到补充后仍将请求分到普通对话。公开负例组在 22 条回放中共预测七次追问。全轮约 97 秒，含从 Windows 入口写入新档案和独立重启加载。核对公开文件、档案、原权重和读出头的摘要。59 项版次单元测试通过。逐题记录位于 `runs/spora-spoken-query-data-v2/result.json`。

## 复用已有的版次识别

为八条新口语题提供正确的查阅许可、人物和事项，仅比较时间坐标。将姓名从残余问句里替换掉，有明确版次时走现有规则，有缺口时再运行本机入口的口语版次头。两边使用同一份档案顺序和相同的原问句。已有口语版次头在这批题上没有补回规则未判定的五条。公开负例组的裸预测覆盖更广，仍会选错三条实际记录，另有一条在只有两版时碰巧选中。此对照只核验可解析的记录编号，不执行读档或开放生成。

| 已知人物和事项 | 版次语义正确 | 解析到正确记录 | 解析到错记录 | 弃权 |
| --- | ---: | ---: | ---: | ---: |
| 新公开负例头的版次预测 | 4/8 | 5/8 | 3/8 | 0/8 |
| 现有显式规则 | 3/8 | 3/8 | 0/8 | 5/8 |
| 现有规则及口语版次头 | 3/8 | 3/8 | 0/8 | 5/8 |

这限制了原样复用。接下来把时间关系独立训练，覆盖「刚换的那一项」「刚换掉的那一项」及「最初选的那一项」等同名同事项的最小对照。用三版以上的档案检验首版、上一版和当前版，分别记录预测的时间关系及实际解析结果。沿用成功的普通请求分流作为对照。TIDER 在候选选择前定位时间范围，StateMemBench 将保留旧值和找出仍有效的值分开评估。这两项研究支持在本机针对版次单测。 [TIDER](https://arxiv.org/abs/2610.23598) · [StateMemBench](https://arxiv.org/abs/2608.19652)

## 复现

公开文件固定为 Xing USP 数据快照 `883c7dc7f52d380b6d4d80067a267ee6921cad39`。下载到本机训练目录，运行器会逐文件校验摘要。 [固定版本](https://huggingface.co/datasets/XingChen-AGI/Xing4.0-29B-Intent-oss/tree/883c7dc7f52d380b6d4d80067a267ee6921cad39/General)

```powershell
$source = 'https://huggingface.co/datasets/XingChen-AGI/Xing4.0-29B-Intent-oss/resolve/883c7dc7f52d380b6d4d80067a267ee6921cad39/General'
$corpus = 'runs/research/xing-usp-883c7dc'
New-Item -ItemType Directory -Force $corpus | Out-Null
foreach ($name in @('smp2020', 'cais')) {
    foreach ($split in @('train', 'test')) {
        $path = "$corpus/$name-$split.json"
        if (!(Test-Path $path)) {
            Invoke-WebRequest "$source/$split/dataset/intent_recognition_${name}_phone_chinese_dataset.json" -OutFile $path
        }
    }
}
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
$out = 'runs/spora-spoken-query-data-new'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_spoken_query_data $corpus runs/spora-spoken-query-plan-span-v1 $out --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_spoken_version_reuse $out "$out/version-reuse.json" --recurrent @weights
```
