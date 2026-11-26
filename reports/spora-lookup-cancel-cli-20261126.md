# 本机入口的查阅取消回放

参考 ToolTurns 的逐轮执行评测，在 Windows 聊天入口中建立查阅范围，发出取消或对照请求，再追问「上一版」，最后重新指定对象与版次。回放使用三版档案的独立副本和第 19、22 层循环注意力候选。逐轮核对实际记录编号、显示值和澄清原因。 [ToolTurns](https://arxiv.org/abs/2611.14925) · [离线选项实验](spora-lookup-backbone-choices-20261126.md)

基线的 9 组、45 轮中，取消或对照当轮通过 7/9。两句独立取消「本次调档无须接着办」和「这轮查阅到此为止」进入了普通聊天。后续「上一版」和重新指定地址均通过 9/9，因为普通聊天当轮也结束了旧的查阅范围。两句取消当轮都没有读档。

修改沿用现有的取消处理，对独立出现的本轮查阅终止指令返回确认，并清除当前范围和未完成的查阅。规则要求整条消息匹配。问句、转述、翻译和附带新任务继续进入原有路径。回放额外加入了待选版次时说「本轮调阅就此结束」。初次修订回放的脚本误将这一组的开场视为已读场景，后续按实际开场类型校验。

此轮验证了已知措辞的入口行为，开放口语取消仍需独立评测。取消只结束当前会话中的查阅。用明确地址重新发起查阅时，仍可以读回原档案记录。

修正后的回放共 10 组、50 轮，取消或对照当轮、后续追问和重新指定对象均通过 10/10。取消后的「上一版」提示补充对象，未读出旧记录。隔离档案完整性检查通过，所有记录与原档案逐条一致。回放约 101 秒，36 项相关测试通过。基线和修正后的逐轮结果分别位于 `runs/spora-lookup-cancel-cli-baseline-v1` 和 `runs/spora-lookup-cancel-cli-v3`。

```powershell
$env:PYTHONUTF8='1'
$weights=@('runs/spora-composition-late-gates-20-23-24-v1/candidate.pt',
           'runs/spora-gelu24-recurrent-layer19-v2','runs/spora-gelu24-recurrent-layer22-v1')
.\.venv\Scripts\python.exe -u -m scripts.audit_spora_lookup_cancel_cli runs/spora-lookup-transitions-clauses-v1 runs/spora-spoken-query-data-v2 runs/spora-lookup-cancel-cli-new --recurrent @weights
```
