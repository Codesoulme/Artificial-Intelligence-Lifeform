"""Assemble the completed persistence and sleep evidence without hiding controls."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def pct(x):return '不适用' if x is None else f'{100*x:.2f}%'


def main():
    output=ROOT/'reports/P9-P10-results.json';document=ROOT/'reports/P9-P10-report.md'
    if output.exists() or document.exists():raise FileExistsError('Retain prior report')
    durable=read('runs/p9-durable-audit/audit.json')
    sleep=read('runs/p10-sleep-audit/audit.json')
    crash=read('runs/p10-sleep-transaction/audit.json')
    profile=read('runs/p10-sleep-profile/profile.json')
    candidate=read('runs/p9-p10-candidates.json')
    for item in (durable,sleep,crash,profile):
        if not item['passed']:raise AssertionError('Incomplete audit')
    if not candidate['source_unchanged'] or candidate['Chinese_training_started']:raise AssertionError('Invalid candidate')
    old=read('reports/P8-read-results.json')['audit']['hashes_before']
    protected={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in old}
    if protected!=old:raise AssertionError('Prior individual or interface changed')
    for name,record in candidate['files'].items():
        if hashlib.sha256((ROOT/'checkpoints'/name).read_bytes()).hexdigest()!=record['sha256']:
            raise AssertionError('Exported candidate changed')
    tests=(ROOT/'runs/p9-p10-final-tests.log').read_text(encoding='utf-8')
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',tests)
    if not match or not tests.rstrip().endswith('OK'):raise AssertionError('Final tests missing')
    development=read('runs/p10-development/results.json')
    rows=[s for c in sleep['chains'] for s in c['stages']]
    control=[v['cost'] for c in sleep['chains'] for v in c['stages'][0]['controls'].values() if 'cost' in v]
    all_training=[x['training'] for x in development]+[x['training'] for x in rows]+control+[candidate['sleep']['training']]
    result={'date':'2026-11-05','parent':'a8b0774','plan':read('configs/p9-p10-plan.json'),
            'development':development,'selection':read('configs/p10-selection.json'),
            'durable':durable,'sleep':sleep,'sleep_transaction':crash,'profile':profile,'candidate':candidate,
            'protected_hashes':protected,'regression':{'count':int(match[1]),'seconds':float(match[2]),'passed':True},
            'consolidation_training':{'optimizer_steps_including_controls':sum(x['steps'] for x in all_training),
                                     'measured_seconds_including_controls':sum(x['seconds'] for x in all_training)},
            'Chinese_training_started':False}
    lines=['# P9/P10：增量保存与睡眠巩固交付','',
           '2026-11-05。完成 SQLite 张量事务、强制进程中断恢复、历史区离线蒸馏及连续学习验收。中文训练仍然暂停，原有个体文件保持不变。','',
           '## 增量提交','',
           '每个条件连续提交 12 次，完整文件与事务路径交替先后执行，表中为中位毫秒数。事务包含 WAL 提交同步及周期性检查点；完整路径包含序列化、检查和落盘。两种安全策略不完全相同，因此不是纯序列化算法排名。','',
           '| 分配段数 | 完整保存 ms | 增量事务 ms | 每次变化张量字节 | 完整文件字节 |','|---|---:|---:|---:|---:|']
    for r in durable['benchmarks']:
        lines.append(f"| {r['pages']} | {r['full_median_ms']:.3f} | {r['incremental_median_ms']:.3f} | {max(r['tensor_payload_bytes']):,} | {r['full_file_bytes']:,} |")
    lines+=['','194,400 字节是本次变化张量的有效载荷，另有约十余 KB 清单、SQLite 页和日志放大。不能据此称实际磁盘写入只有这个数。8/16 段条件包含空区域，未扩展记忆正确性的实测范围。','',
            '| 分配段数 | 含日志/共享文件，整理前字节 | 截短日志后字节 | 最终检查点 ms |','|---|---:|---:|---:|']
    for r in durable['benchmarks']:
        lines.append(f"| {r['pages']} | {r['database_with_wal_before_checkpoint']:,} | {r['database_with_wal_after_checkpoint']:,} | {r['checkpoint_ms']:.3f} |")
    lines+=['','日志可复用已分配空间，但磁盘文件不一定立即缩小；备份也另占空间。完整读取目前仍加载全部个体，尚未改为磁盘按需加载。','',
            '## 崩溃恢复','',
            '普通写入与睡眠替换各在七个位置强制结束工作进程，共十四次。所有结果均为完整旧状态或已提交新状态；并发读者没有看到半次更新，后续写入、备份恢复及整理结果一致。','',
            '| 中断位置 | 普通观察恢复 | 睡眠替换恢复 | 恢复后下一次学习 |','|---|---|---|---|']
    for ordinary,sealed in zip(durable['crashes'],crash['cases']):
        lines.append(f"| {ordinary['stage']} | {'通过' if ordinary['exact'] else '失败'} | {'通过' if sealed['exact'] else '失败'} | {'一致' if ordinary['next_write_exact'] and sealed['next_write_exact'] else '失败'} |")
    lines+=['','这证明所测进程中断情形的事务一致性，不是真实断电、硬件故障或所有并发场景的保证。旧版本写者拒绝、篡改检测和失败回滚另有回归覆盖。','',
            '## 睡眠的实现与结果','',
            '保持骨干、输入坐标及当前矩阵不动，将已满历史段的神经读出蒸馏到随机初始化的低秩因子；当前选中 rank16、600 步。辅助逆相关矩阵通过对称性精确保存上三角。只有检查通过，才在新状态中移除该段旧密集表示。','',
            '开发比较了三种秩与解析 SVD 对照，随后锁定选择文件。新经历流中三个随机底座、每个两个世界、各三次转存，共六条链和十八次睡眠全部通过。世界流跨底座复用，不能把它们称为六套完全独立环境。','',
            '| 底座种子 | 世界 | 最低历史 | 最低早期历史 | 最低不可信历史 | 最低错误组合拒答 | 最低当前值 | 后续纠正 | 独立进程恢复 |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in sleep['chains']:
        stages=c['stages'];corrections=[x for r in stages for x in r['corrections']]
        lines.append('| '+str(c['seed'])+' | '+str(c['world'])+' | '+' | '.join(pct(min(s['after'][k] for s in stages if s['after'][k] is not None)) for k in ('history','oldest64','untrusted_history','invalid_unknown','current'))+f" | {sum(corrections)}/{len(corrections)} | {sum(s['cold_exact'] for s in stages)}/3 |")
    agreement=min(g['agreement'] for r in rows for g in r['gate']['groups'].values() if g['agreement'] is not None)
    difference=max(r['gate']['max_probability_difference'] for r in rows)
    lines+=['',f"完整有限域的教师已知/未知分组中，最低类别一致率为 {pct(agreement)}；最大概率差为 {difference:.4f}。因此不宣称概率分布不变。实际已发生历史、当前值与继续纠正的结果见表，拒答仍有少量错误。",
            '', '训练探针遍历 64 键×64 局部时刻，并学习教师读出；检查相同有限域属于记忆转移评估，不是未见事实泛化。无文本语义、来源判断或世界模型训练。',
            '', '## 压缩与控制','',
            '每段状态从 32,400 降为 13,950 个标量，减少 56.94%。读出矩阵的低秩化贡献 10,440 个标量，辅助状态对称打包贡献 8,010 个标量。只有 16 种结果的有限任务有利于低秩表示，开放经历未验证。','',
            '| 控制 | 首次转存有效历史 | 容量变化 | 解释 |','|---|---:|---|---|',
            '| 不转存便清空 | 0% | 移除历史 | 回忆依赖内部载体 |',
            '| 同秩 SVD | 100% | 与学生相同 | 强解析基线也成功，未证明训练优于它 |',
            '| 等步数全矩阵重演 | 100% | 不减少 | 步数相同但计算量不同，非严格等算力在线对照 |',
            '| 随机学生蒸馏 | 100% | 每段减少56.94% | 完成神经转存与后续学习验证 |','',
            '控制在每条链第一段进行，没有把对照的单段成绩外推到它已验证三段。当前通道不参与压缩，后续纠正通过不能证明共享快慢通路的所有冲突已解决。',
            '', '## 完整个体资源','']
    f=profile['files']
    lines += [f"三段封存、第四段待用：活动及封存状态总量 {f['dense_scalars']['total']:,} → {f['sealed_scalars']['total']:,}，减少 {pct(1-f['sealed_scalars']['total']/f['dense_scalars']['total'])}。",
              f"完整便携文件 {f['dense_bytes']:,} → {f['sealed_bytes']:,} 字节，减少 {pct(1-f['sealed_bytes']/f['dense_bytes'])}；整理后的 SQLite 文件 {f['dense_sqlite_bytes']:,} → {f['sealed_sqlite_bytes']:,} 字节。固定骨干仍占主体，不能称整个模型缩小57%。",
              '', '训练优化器和教师探针是临时空间；父快照保留在本地以便回滚，没有从总磁盘占用中消失。下表包含读取选中封存段时的重建成本，计时交错执行。','',
              '| 设备 | 查询规模 | 原密集或同内容展开 ms | 封存状态 ms |','|---|---|---:|---:|']
    for row in profile['rows']:
        values=list(row['timing'].values())
        lines.append(f"| {row['device']} | {row['query_count']} | {values[0]['median_ms']:.3f} | {values[1]['median_ms']:.3f} |")
    lines+=['','封存状态在部分条件更慢。当前实现用存储换重建计算；字节接口还会展开各段并重建读取计划，未解决压缩后所有运行内存和延迟问题。这是下一步可优化的边界，不把本轮压缩称为推理提速。',
            '',f"所有开发、独立转存、全矩阵控制和候选导出合计 {result['consolidation_training']['optimizer_steps_including_controls']:,} 次优化步；计入的实测转存及控制耗时 {result['consolidation_training']['measured_seconds_including_controls']:.2f} 秒，不含审计、数据持久化和进程启动。具体每步乘加估算、探针数及完整费用在 JSON。",
            '', '## 交付个体与完整性','',
            '- individual-durable.sqlite 精确保留原三次经历，原 individual-prelanguage.pt 不变。',
            '- individual-sleep-candidate.sqlite 从这一谱系继续，追加69次程序生成观察，共72次；第一段睡眠后仍继续学习。它不是新随机底座。',
            '- 对应便携文件为 individual-sleep-candidate.pt；当前12→7，最初历史→5，第二次→9，不可信第三次→3。',
            '- 三段压缩的资源数字来自审计链；交付候选只有第一段封存，不能混为同一压缩比例。',
            f"- {result['regression']['count']} 项回归通过；六份原有个体/接口摘要不变；中文训练量为零。",
            '', '结构、命令和边界见 [实施说明](../docs/durable-sleep-implementation.md)。本轮完成的是持久事务与有限神经记忆转存，世界模型想象、自生成课程和通用语义抽象仍未实现。']
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    document.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(document),'tests':result['regression']['count'],'Chinese_training_started':False},ensure_ascii=False))

if __name__=='__main__':main()
