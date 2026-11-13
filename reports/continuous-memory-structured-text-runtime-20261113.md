# 主运行时结构化连续文本记忆

## 变更

主 `SporaChat` 入口新增显式研究接口：

- `observe_structured_text(key, identity_text, fact_text)`：把身份和事实分 lane 编码后写入孢子状态。
- `read_structured_text(key)`：从当前状态解码 token 和文本，不把记忆 tensor 暴露给调用方。
- codec 的 seed 纳入持久化配置，并支持从配置重建，避免重载时使用未记录的随机投影。

接口要求调用者提供已经验证的地址和身份/事实边界；自然语言路由不在这个底层原语中。

## 真实运行时验证

在隔离 DeepSeek/SPORA working 个体上：

1. 写入身份 `小梅`，事实 `喜欢喝茶`。
2. 关闭第一进程。
3. 独立进程重新加载同一 working checkpoint。
4. 读取身份和事实 token，并与写入 token 逐项比较。

结果：`passed: true`、`same_checkpoint: true`、identity `true`、fact `true`，结构化 revision 为 `1`。写入前后 working checkpoint SHA-256 均为 `37ceb81db79b73f4b6925bad5ec4dfb114606c8026c26ab0018c2b1db668d80b`。

## 判定

结构化连续文本通道已经进入主运行时，并具备增量保存和跨进程恢复能力。这是任意受容量约束的中文 token 记忆基础，但还不是开放中文自动写入或自由聊天：地址、身份/事实分段仍由调用者提供，回答生成尚未把该通道接入 Qwen 对话解码路径。

## 原始产物

- `runs/deepseek-spora-structured-text-read-after-write-20261113/write.json`
- `runs/deepseek-spora-structured-text-read-after-write-20261113/result.json`
- `scripts/deepseek_spora_structured_text_read_after_write_audit.py`
