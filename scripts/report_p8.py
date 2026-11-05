"""Publish measured read costs, unsuccessful shortcuts and integration boundaries."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    destination = ROOT/'reports/P8-read-results.json'
    document = ROOT/'reports/P8-read-report.md'
    if destination.exists() or document.exists():
        raise FileExistsError('Preserve previously published evidence')
    audit = read('runs/p8-read-audit-v3/audit.json')
    profile = read('runs/p8-paired-profile.json')
    first = read('runs/p8-read-audit/audit.json')
    if not audit['passed'] or not profile['passed']:
        raise AssertionError('Read validation incomplete')
    for name, expected in audit['hashes_before'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise AssertionError('Protected individual or interface changed')
    test = (ROOT/'runs/p8-final-tests.log').read_text(encoding='utf-8')
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s', test)
    if not match or not test.rstrip().endswith('OK'):
        raise AssertionError('Full regression required')
    sources = read('runs/p8-source-verification.json')
    maximum = max(row['max_abs_logit_error'] for case in audit['end_to_end'] for row in case['results'].values())
    result = {'date': '2026-11-05', 'parent_commit': 'b59ee9a', 'audit': audit,
              'interleaved_profile': profile, 'initial_measurement': first,
              'regression': {'tests': int(match.group(1)), 'seconds': float(match.group(2)), 'passed': True},
              'sources': {name: {'url': value['url'], 'verified': value['ok']} for name, value in sources.items()},
              'intermediate_run': {'path': 'runs/p8-read-audit-v2', 'complete': False,
                                   'note': 'Process ended before the final report; excluded from pass counts. Final v3 completed.'},
              'Chinese_training_started': False, 'persistent_store_replaced': False,
              'sleep_integrated_into_new_architecture': False}
    lines = ['# P8：读取优化与后续记忆系统安排', '',
             '2026-11-05。本轮没有训练中文，没有更新底座或候选经历。实现等价的读取合并与可选的有界筛选，核对睡眠机制，并给出持久化实验建议。', '',
             '## 已启用与尚未启用', '',
             '- 默认 exact：相同矩阵只算一次，段的位置权重仍完整参与。相同空段多时节省明显；全部不同的段仍需完整计算。',
             '- 可选 bounded：依据神经矩阵导出的分数范围，只在遗漏误差可被限制时挑选少量段，否则完整读取。没有训练语义路由。',
             '- 当前随机接口注意力分散，真实个体上的 bounded 通常没有筛掉段，还增加判断开销。因此没有把它设为默认。',
             '- 训练保持旧可微路径；新经历和扩容会使读取计划失效并重建。计划占用额外内存，不会缩小模型文件。',
             '- 总体设计的睡眠巩固尚未合入新历史结构，持久存储格式也尚未替换。这两项有了具体的下一轮验收顺序。', '',
             '## 输出保持与反例', '',
             f"48 个既有审计个体分别在 CPU 和 GPU 上运行，共 {len(audit['end_to_end'])} 个设备/个体组合；使用随机字节，默认和可选路径的第一候选均与旧路径一致。最大输出分数偏差 {maximum:.3g}，门槛为 1e-4。这不是语言正确率或新的记忆容量成绩。",
             '', '| 数值条件 | 查询数 | 通过上界检查而减少读取 | 最大读出 L2 偏差 |',
             '|---|---:|---:|---:|']
    labels = {'ambiguous': '权重分散', 'selective': '单段贡献集中',
              'misleading': '位置偏好与矩阵内容冲突', 'random_selective': '随机方向、较强选择性'}
    for case, label in labels.items():
        rows = [x for x in audit['read_tests'] if x['case'] == case]
        lines.append(f"| {label} | {sum(x['stats']['queries'] for x in rows)} | "
                     f"{sum(x['stats']['pruned_queries'] for x in rows)} | {max(x['max_l2_error'] for x in rows):.3g} |")
    lines += ['', '这些数值构造覆盖三个新随机种子及不同段数，CPU/GPU 重复不应被称为独立现实样本。单元测试另含非对角信息误导、原位更新、后续观察、扩容与梯度保护。误差上界针对一次记忆读取，不构成任意后续网络的最终类别保证。', '',
              '## 交错计时', '',
              '每种路径先预热三次，再轮换执行顺序，各测 21 次；下表为完整字节接口前向的中位毫秒数。已包含路由检查和回退，未包含启动、加载、第一次构建计划或保存。单位置也不是一整个聊天回复。机器有其他桌面负载，不提供硬实时保证。', '',
              'candidate 为三次演示经历，occupied4 为已经过 256 次观察的四段状态。8/16 段仅追加空区域；没有声称它已学满 1,024 次经历。', '',
              '| 设备 | 状态 | 段数 | 批量×长度 | 原路径 ms | 默认 ms | 筛选 ms | 默认加速比 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in profile['rows']:
        times = row['timing']
        lines.append(f"| {row['device']} | {row['layout']} | {row['pages']} | {row['batch']}×{row['length']} | "
                     f"{times['legacy']['median_ms']:.3f} | {times['exact']['median_ms']:.3f} | "
                     f"{times['bounded']['median_ms']:.3f} | {row['exact_speedup']:.2f}× |")
    regression = [x for x in profile['rows'] if x['exact_speedup'] < 1]
    lines += ['', f"本次交错计时有 {len(regression)} 个条件默认路径更慢。小容量调度开销和运行波动都可能抵消收益，不能概括为所有条件均加速。初测还出现 GPU 输出耗时大于预期，随后去掉不必要的同步、减少展开临时张量，并用上述交错计时复核；初测数字完整保留在 JSON。",
              '', '中间 v2 进程在最终报告生成前结束，未捕获到完整异常说明，不计入通过证据；最终 v3 完成全部条件，另有独立计时及全量回归。', '',
              '## 读取数量及额外开销', '',
              '16 段 candidate 加上当前矩阵，原来每个位置每层读 17 次；默认合并后读 3 个不同矩阵，重型运算减少 82.35%。occupied4 扩至相同容量则从 17 次降至 6 次。位置权重仍逐段参与，不能把这称作总成本 O(1)。', '',
              '| 设备 | 状态 | 段数 | 单层计划构建 ms | 单层计划张量 KiB（含可选界限） |',
              '|---|---|---:|---:|---:|']
    for row in audit['benchmarks']:
        lines.append(f"| {row['device']} | {row['layout']} | {row['pages']} | {row['prepare_ms']:.3f} | "
                     f"{row['plan_tensor_bytes_after_bounds']/1024:.2f} |")
    lines += ['', '计划只从模型内部矩阵导出，不存原始事件或答案。额外开销按层计，表中不含 Python 对象和源张量本身；首次 bounded 调用还会构建范数界限，预热后计时不包含这一成本。持续更新会触发重建，不能把稳态结果外推到每个输入都改变记忆的场景。', '',
              '## 做梦与持久保存', '',
              '原设计第 9 节是回放辅助的长期巩固，第 11.3 节是世界模型想象。P3 证明过旧结构上的部分转存及再次纠正，但不是新版的历史压缩证明。现在可以先在副本上做新架构的符号巩固试验；清理快区域之前，必须通过屏蔽原区后的回忆、纠错、未知判断与后续学习检查。想象轨迹还需要世界预测能力。',
              '', '下一轮存储建议优先比较 SQLite 事务内更新张量块与现有完整保存，再考虑 safetensors 封存快照。数据库负责原子提交和恢复，回答仍由神经运算产生。RocksDB 可作为规模扩大后的备选。没有在本轮迁移任何个体。',
              '', '具体机制、研究出处和实施顺序见 [结构与研究补充](../docs/memory-read-sleep-storage.md)。', '',
              '## 验收与停止点', '',
              f"- {result['regression']['tests']} 项回归通过，耗时 {result['regression']['seconds']:.3f} 秒。",
              '- 三份早期个体、新候选、父个体与原始字节接口的 SHA256 全部保持。',
              '- 中文训练量仍为零。没有下载中文语料，没有启动巩固或替换存储。',
              '- 可复核脚本：scripts/audit_memory_reads.py、scripts/profile_memory_reads.py、scripts/report_p8.py。',
              '- 已实现计算层面的按需读取与保守回退；理解问题后唤起相关往事的语义能力仍未学习。']
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    document.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'report': str(document), 'regression_tests': result['regression']['tests'],
                      'Chinese_training_started': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
