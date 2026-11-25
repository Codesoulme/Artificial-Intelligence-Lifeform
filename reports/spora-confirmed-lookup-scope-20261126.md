# 复用已确认的查阅地址与补全缺项

前轮的定向追问会漏问真正缺失的坐标，也会对已经提供的姓名重复追问。本轮分开验证两种可用证据：本轮措辞中的姓名与事项，以及上轮实际读取所得的单一地址。*Lost with a Map* 区分状态是否可辨认、是否解析出正确的值，以及动作是否采用了该值。ClariEdge 将可由前文解决的请求作为难负例。StateGraph 按前提是否仍有效撤销依赖状态。 [前轮实验](spora-spoken-coordinate-clarification-20261126.md) · [Lost with a Map](https://arxiv.org/abs/2609.33883) · [ClariEdge](https://arxiv.org/abs/2610.28686) · [StateGraph](https://arxiv.org/abs/2611.02912)

## 跨轮读取

本机入口支持明确指向「刚才查到的那项」且要求重新查阅的短句。有一个已确认地址时使用该地址。用户指定版次就直接读取，缺版次时只问版次。已查过两项、转题、取消、清空或重启时，入口询问姓名和事项。若这一问已明确版次，补答可以只给地址。未完成的普通查阅不能作为「刚才查到」的依据。

使用独立 Windows 进程回放，档案从一份已有四条三版时间线的张量档案复制。逐条检查实际读取编号、输出中的登记值、是否需要追问及档案完整性。一次对罗映禾地点的当前版读取选中「枫桥亭」，改查上一版选中「望湖台」，随后只补「首版」选中「海棠廊」。切换到罗映荷的饮品后，同类指代选中罗映荷的首版「山楂茶」。重启后，入口先问地址，补上「穆云川的用品」时沿用该问题已明确的「上一版」，读回记录 8 的「红绳」。

首次启动 20/20，重启后 4/4，分别耗时约 44 秒和 41 秒。档案完整性检查通过，原档案摘要与逐条记录保持一致。66 项版次单元测试通过。逐题结果保存于 `runs/spora-confirmed-lookup-scope-v3/result.json`，组合实验保存于 `runs/spora-coordinate-fusion-v1/result.json`。

## 两个冻结查阅头的组合

用同一份 16 条记录档案回放 24 条新写的口语题。四对请求比较给出人物和省略人物的情形，另有明确提两个人、只提干扰人物、缺事项、取消和改写等情形。模型使用原有稠密地址头。两组冻结的动作头沿用原阈值。直接组合先采纳定向头，未提议时再用整地址头。证据一致组在此基础上允许一个覆盖条件：定向头认为缺人，但唯一被提到的姓名、定向头给出的事项和整地址头的建议三者一致时，提议确认完整地址。实际读取需经过用户模拟补答或确认。错误提议收到取消。

| 指标 | 整地址提议 | 定向追问 | 直接组合 | 加入一致性条件 |
| --- | ---: | ---: | ---: | ---: |
| 缺姓名，补答后读对 | 0/6 | 1/6 | 1/6 | 1/6 |
| 缺事项，补答后读对 | 0/4 | 2/4 | 2/4 | 2/4 |
| 地址已足够，读对 | 2/7 | 2/7 | 3/7 | 3/7 |
| 应查题提议有误 | 2 | 0 | 1 | 1 |
| 非查阅请求中的多余追问 | 0/7 | 0/7 | 0/7 | 0/7 |
| 读错记录 | 0 | 0 | 0 | 0 |

两种组合多恢复了一条已给定完整信息的请求，并把「夏宁遥那杯我已经买了。另一杯还不知道选哪种，具体是谁的我再说」误提议为查夏宁遥。唯一出现的姓名只是已处理对象，不能直接当作待查对象。一致性条件在这批题中未改变结果。约 74 秒完成。本轮样本由项目内撰写，仅用于诊断，两个动作头没有重新训练。

下一轮优先检验对候选的用途判断，让模型从成对训练样本中区分真正要查的人、已排除的人和待补姓名。固定版次档案，比较三种情形的正确追问、错提议和补答后读取。 [实例归属评测](https://arxiv.org/abs/2610.34343)

## 复现

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:PYTHONUTF8 = '1'
$weights = @('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
             'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_confirmed_lookup_scope runs/spora-spoken-temporal-head-v1 runs/spora-confirmed-lookup-scope-new --recurrent @weights
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_coordinate_fusion runs/spora-read-confirmation-action-v1 runs/spora-spoken-coordinate-clarification-v2 runs/spora-crosswoz-field-transfer-v1 runs/spora-dense-joint-address-v1 runs/spora-coordinate-fusion-new --recurrent @weights
```
