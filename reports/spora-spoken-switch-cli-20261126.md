# Windows 聊天入口的改查与确认

上轮用实验给定的新旧地址验证了改查后的版次读取。本轮把明确的「改查」「换查」接到现有口语查阅入口。上一轮已确认一项时，收到明确改查会清除旧地址。新地址和版次写全时走现有读回。写清地址但缺版次时询问这项的版次。地址未确定时尝试现有口语模型，只有用户确认提议的地址和版次后才读取。模型无法提议地址时，入口询问姓名和事项。 [上轮读回实验](spora-revocation-readback-20261126.md) · [结构化解析研究](https://arxiv.org/abs/2610.34343) · [ToolTurns](https://arxiv.org/abs/2611.14925)

Windows 入口使用从三版实验档案备份出的独立档案，完成 7 组、40 轮回放。逐轮核对实际编号和可见值，档案内容在回放后与源档案逐条一致。姓名、事项、版次都齐备的改查直接读对上一版「潮汐」。信息缺版次时先问版次，用户答「上一版」后读对「桃源路」。新对象未知或给出多个选择时先追问，取消后孤立的「首版」仍会追问对象，未读取旧档。

三个写成「碰头」「拿的东西」「后来改喝哪种」的请求里，口语模型正确提议地址为 0/3，入口都请用户补充地址。首次回放在取消后发送的完整控制语句缺少明确的「请查」，这些控制未触发查阅。修正回放后，三个显式控制均读对。模型提出新地址、等用户补齐后才实际读取的路径由单元测试覆盖。125 项相关测试通过，Windows 复跑约 134 秒。

改查入口的状态更新已通过本机验证。下一步以这批口语失败作为回归对照，在独立措辞上检验候选地址排序，分别报告正确提议、错提议和追问后的实际完成数。

```powershell
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2', 'runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_spoken_switch_cli runs/spora-lookup-transitions-clauses-v1 runs/spora-spoken-query-data-v2 runs/spora-spoken-switch-cli-new --recurrent @weights
```

逐轮结果位于 `runs/spora-spoken-switch-cli-v2/result.json`。
