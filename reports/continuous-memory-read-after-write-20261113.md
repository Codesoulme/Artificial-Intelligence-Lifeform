# 跨进程中文写入后读取审计

## 目的

验证自然语言观察入口完成一次事实更正后，关闭当前进程、重新加载隔离个体，读取结果是否仍保持一致。

## 测试

- 基线个体当前事实：小梅偏爱的饮料为“咖啡”。
- 第一进程提交：`小梅偏爱的饮料是茶。`
- 第一进程读取：接受，返回“茶”。
- 第二进程独立重载同一 working checkpoint 后读取：接受，返回“茶”。
- 写入前后 checkpoint SHA-256：`5ea8a828145ddd5d464f7c88ec94da279a1068082950257f94a9c81752916c18`。

## 结果

通过。动作是 `correct`，重载后 `same_answer=true`、`same_checkpoint=true`。

## 边界

这验证的是隔离个体、有限饮料词表和研究用自然写入头的持久化闭环；它不证明开放世界中文解析、自由聊天或任意事实类型都已可靠。默认聊天入口仍保持显式 opt-in。

## 原始产物

- `runs/deepseek-spora-read-after-write-audit-v2-20261113/write.json`
- `runs/deepseek-spora-read-after-write-audit-v2-20261113/result.json`
- `scripts/deepseek_spora_runtime_read_after_write_audit.py`
