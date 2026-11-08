# P88：Linux 101 固定版本的本地获取与抽样

2026-11-08。依据 P87，仅用于来源和质量研究，**未批准训练，也未进行训练**。源码及样本均存放在被 Git 忽略的 `runs/research/`；这里仅记录统计和方法。

来源为 [ustclug/Linux101-docs](https://github.com/ustclug/Linux101-docs)，提交 `4099db4a1f90e8e89cac3eca4bf2e36b2932e3f1`，根树 `43410def119131fbd5ec1e5f1960c02b450a8817`。获取 `docs/` 下 35 个 Markdown，共 484,174 字节，另取根目录 `LICENSE`（20,131 字节）和 `README.md`（1,030 字节），总计 37 个文件、505,335 字节，均小于 1,000,000 字节。字节数包含标记、命令等，并非自然语言数量。

获取时 GitHub 返回的递归树声明 `truncated: false`；脚本依据所有树项重算每层 Git tree 对象哈希，并与固定提交所指根树对照。每个选中文件按树元数据核对长度与 Git blob SHA-1，另记录 SHA-256、路径、仓库和版本；原始树元数据、清单、完整源文件与原有署名均保留在 `runs/research/linux101-4099db4a/`。树对象哈希本身不包含文件长度，因此长度另与实际下载逐一对照。元数据经 HTTPS 从 GitHub 获取，不构成独立签名验证。

本机默认请求因吊销检查不可用而失败，实际使用了 `--ssl-no-revoke`；证书链校验保留，但未完成证书吊销检查。脚本拒绝不安全路径、非常规选中对象和既有输出目录；失败后保留的部分目录须人工检查，并用新目录名重新运行。

使用脚本中的启发式筛选，在按路径与行号顺序排列的候选中以 Python `random.Random(87).sample` 抽取 20 段；共 570 个候选。样本仅在本地 `runs/research/linux101-sample-v2-seed87/samples.json`，每段包含源文件、起止行、保留换行的原文、原文 UTF-8 SHA-256 及代码和图表依赖提醒。筛选排除围栏代码、缩进代码以及含反引号的文本块，也跳过部分 Markdown 结构；并非完整 Markdown 解析器，未标记的代码或其他标记仍可能出现，且段落有可能依赖邻近文字、示例、图或表，不能认定上下文完好。候选数量是筛选结果，不等同于质量或权利合格数量。

模型审阅者对照原文件邻近上下文逐条检查这 20 段，1 段初判可作有限候选、7 段存疑、12 段排除。许多片段依赖后续列表、截图或命令，也有时效性和范围不明的问题；此结果不支持把机器筛出的段落整体用于训练。此次不是人类专家评审，亦未验证外部链接或现行技术事实；样本量小，不代表总体精确比例。

项目 README 声明文档使用 CC BY-SA 4.0；保留的许可文件和原文有助于核对归属。贡献者、引用、图片、截图、代码及其他第三方材料尚未逐项核实，署名、相同方式共享及拟议用途的要求须另行评估；公开声明不等于本项目训练批准。

复现时从仓库根目录执行，输出名称须是未存在的目录：

```powershell
python scripts/linux101_research.py fetch --output linux101-new --ssl-no-revoke
python scripts/linux101_research.py sample --source linux101-new --output linux101-sample-new --seed 87 --count 20
python -m unittest discover -s tests -p test_linux101_research.py -v
```

若环境吊销检查正常，应省略 `--ssl-no-revoke`。抽样可复现性依赖相同源码、脚本及 Python 随机实现；记录的摘要可用于核对。五项测试覆盖树截断与篡改、路径和覆盖、代码筛除、文件改动检测及固定种子抽样。
