# P166 TinyStories 过滤后候选包

根据 P165 复核结果，排除 7 条 major，生成 57 条 clean/minor 故事候选。候选包绑定：

- Gabrui 多语言 TinyStories 固定版本；
- 首个训练分片 SHA-256；
- P164 队列摘要；
- P165 复核摘要；
- 项目 tokenizer 摘要；
- CDLA-Sharing-1.0 声明及原始来源署名链。

候选状态明确为 `candidate_not_approved_for_training`，没有训练批准，也没有修改原分片。57 条故事只适合后续极小连续文本诊断，不能承担通用中文预训练。若要训练，必须重新确认复核独立性、故事族隔离、旧训练/开发隔离、实际目标规模和受限预算。
