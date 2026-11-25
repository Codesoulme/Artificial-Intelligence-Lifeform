# 多事项档案中的联合地址和弃权

上轮只允许真实目录中的地址，八条口语题选对五条。给每个人补齐四种事项后，正确数回到两条。本轮把候选密度做进真实档案，训练同时选择人物与事项的小头。Xing USP 和 CalOn 将实体归属与类别分开评估，读取许可研究也要求区分有效证据与不合格候选。 [上轮目录对照](spora-spoken-address-authority-20261126.md) · [Xing USP](https://arxiv.org/abs/2610.34343) · [CalOn](https://arxiv.org/abs/2610.17113) · [读取许可验证](https://arxiv.org/abs/2610.07309)

## 设置

加载当前两层循环版，冻结前 12 层作为特征来源。联合头为至多两位被提及的人物和四类事项打分，另有一个「无匹配」输出。训练时，同一正例既出现在完整目录中，也出现在移除正确地址的目录中。前者监督精确地址，后者监督无匹配。全局候选和局部候选使用同样的小头及训练设置。局部方案还输入围绕姓名出现位置的短片段，检验是否有助于区分相邻事项。训练中也加入公开普通请求、同句撤回和编辑样本。

用常规 Windows 入口从空档案写入四位新人物，每人各有饮品、见面地点、用品、书目四条记录。两对姓名只有末字不同。写完退出，重启后评测八条口语查阅与四条无需查阅的请求。评分核对实际地址、读取编号和读回的张量值。版本条件固定为读取现行记录。独立校准中的姓名和句式与训练及留出分开。移除正确地址的对照仅撤去候选列表中的目标，观察是否误选剩余地址，原始档案保持原样。

## 查阅和弃权

训练使用 384 条原始请求，给每条查阅正例加入缺失目标的对照后得到 512 个训练视图，校准使用 48 条。约 83 秒完成写入、重启、训练与读回。两种联合头在八条应查题的最高分结果均为无匹配。即使接受任意最高分地址，也不会恢复这八次查阅。

| 指标 | 上轮地址头 | 联合头，全句 | 联合头，含姓名附近片段 |
| --- | ---: | ---: | ---: |
| 实际读对 | 0/8 | 0/8 | 0/8 |
| 12 条回放中的实际错读 | 1 | 0 | 0 |
| 移除正确地址后弃权 | 7/8 | 8/8 | 8/8 |
| 人工给定许可后选址正确 | 0/8 | 3/8 | 3/8 |

给定许可一行只在地址候选间排序，记录不取值。两个联合头确实学会了少量联合寻址，但识别新问法的覆盖不够。另用完整循环版解码器按编号选择地址。只提供姓名和事项，允许回答零。正常顺序答对 1/8，倒序并重新分配编号后答对 3/8，八条中没有一条两次都选对。移除目标的对照两种顺序都未能正确弃权。解码用时约 42 秒，读到的只是目录元数据。 [隐藏表示与输出对照](https://arxiv.org/abs/2610.27455)

回放确认了 16 条真实张量记录和原档案摘要。62 项版次测试通过。训练及诊断文件分别为 `runs/spora-dense-joint-address-v1/result.json` 和 `runs/spora-dense-joint-address-v1/decoder.json`。

## 公开正例准备

当前的动作和地址监督主要来自自编模板。核对 CrossWOZ 的官方训练、验证和测试档案后，按用户回合的 `Request` 标签提取域与请求字段。有多个字段或前后请求域切换的回合进入清单，清单保存来源会话、回合编号、字段及原句摘要。三个官方分区的会话编号互不重叠，数据压缩包摘要与固定版本一致。独立重跑的三份清单和总表逐字节一致。 [CrossWOZ 论文](https://aclanthology.org/2020.tacl-1.19/) · [固定数据版本](https://github.com/thu-coai/CrossWOZ/tree/df82c9fdff91b9b130f2d6b89110d3870ba6260e/data/crosswoz)

| 官方分区 | 多字段请求 | 与上一请求域完全不同 | 实际清单行数 |
| --- | ---: | ---: | ---: |
| 训练 | 5,848 | 10,119 | 13,473 |
| 验证 | 573 | 1,018 | 1,343 |
| 测试 | 589 | 1,040 | 1,360 |

同轮跨域只有四条训练样本，验证与测试均为零。它适合训练自然中文的字段选择和跨轮任务归属，跨人记忆仍用独立档案检验。下一轮先在这些真实正例上训练，并分开报告任务字段和个人档案寻址的泛化。清单位置为 `runs/research/crosswoz-positive-fields-v1/`。

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
$out = 'runs/spora-dense-joint-address-new'
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_dense_joint_address runs/spora-spoken-proposal-v1 runs/research/xing-usp-883c7dc $out --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_dense_address_decoding $out "$out/decoder.json" --recurrent @weights
```

```powershell
$corpus = 'runs/research/crosswoz-df82c9f'
$source = 'https://raw.githubusercontent.com/thu-coai/CrossWOZ/df82c9fdff91b9b130f2d6b89110d3870ba6260e/data/crosswoz'
New-Item -ItemType Directory -Force $corpus | Out-Null
foreach ($split in @('train', 'val', 'test')) {
    $path = "$corpus/$split.json.zip"
    if (!(Test-Path $path)) { Invoke-WebRequest "$source/$split.json.zip" -OutFile $path }
}
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_crosswoz_positive_fields $corpus runs/research/crosswoz-positive-fields-new
```
