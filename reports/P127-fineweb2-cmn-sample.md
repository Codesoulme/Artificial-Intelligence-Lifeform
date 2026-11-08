# P127 FineWeb2 中文小分片审计

对固定 revision `af9c13333eb981300149d5ca60a8e9d659b276b9` 的 `cmn_Hani/train/000_00000.parquet` 第 0 个 row group 做了 HTTP Range 审计，只读取约 10.5 MiB，得到 1,000 条记录；未下载整分片。

该 row group 的结构提供 `text`、URL、日期、语言、语言分数、脚本和 MinHash 簇大小。1,000 条均标为 `cmn/Hani`，但来源域很杂，包含旅游平台、Wikipedia、博客、医疗/国际机构、新闻和商业网站。正则筛查命中 41 个邮箱模式和 7 个手机号模式；这些是规则命中，不是个人信息确证，但足以说明需要逐域隐私和内容审查。

该样本不能代表全分片，也没有完成语义质量、版权或跨语料隔离审查。因此 FineWeb2 仍是待审候选，未生成训练数据或训练批准。正文样本仅保存在被 Git 忽略的 `runs/p127-fineweb2-cmn-sample-20261109`。
