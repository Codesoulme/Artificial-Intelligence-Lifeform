# 本地语义记忆交流入口

新增 `scripts/deepseek_spora_semantic_chat.py`，使用当前已验证的序列语义槽和独立状态文件：

```powershell
Get-Content .\conversation.txt -Encoding UTF8 | .\.venv\Scripts\python.exe scripts/deepseek_spora_semantic_chat.py --state D:\SPORA\runs\semantic-chat-state.pt
```

运行时输入形如“某人喜欢喝某值”的事实会写入状态并原子保存；输入“某人喜欢喝什么饮料？”会按人物和关系读取完整值；未知人物不读取其他人的槽；不属于受支持记忆句式的问题交给底座普通生成。

UTF-8 冒烟结果：写入“小赵喜欢喝椰汁”后查询返回完整“椰汁”，查询未记录的小吴返回“尚未记录”提示。状态文件可在下一次进程启动时加载，底座摘要和语义接口绑定会校验。

当前入口仍是研究原型：只支持有限中文句式和饮品/颜色关系，写入使用底座 token embedding，完整自由聊天、身份消歧、冲突处理和睡眠巩固尚未完成。可学习隐藏写入器的广覆盖实验也未通过，因此入口暂时使用已经通过多 token 与重载验收的直接文本编码。
