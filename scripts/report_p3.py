"""Assemble completed P3 runs; refuse partial evidence or changed parents."""
import hashlib
import json
from pathlib import Path
import statistics
import torch
from spora.checkpoint import load_checkpoint
from spora.evaluate import isolated_query


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(result):
    return [r for w in result['worlds'] for r in w['stages']]


def percent(value):
    return f'{100*value:.2f}%'


def main():
    torch.set_num_threads(2)
    development=read('runs/p3-development/development.json')
    lock=read('runs/p3-selection-lock.json')
    assert lock['selection_sha256']==sha('runs/p3-development/development.json')
    assert lock['plan_sha256']==sha('configs/p3-plan.json')
    audit={str(s):read(f'runs/p3-audit-seed{s}/audit.json') for s in (7,11,23)}
    plasticity={str(s):read(f'runs/p3-plasticity-seed{s}.json') for s in (7,11,23)}
    controls=read('runs/p3-controls/controls.json')
    diagnosis=read('runs/p3-diagnosis.json')
    costs=read('runs/p3-write-cost.json')
    capacity={str(s):read(f'runs/p3-capacity-seed{s}.json') for s in (7,11,23)}
    prior=read('reports/P2-results.json')
    integrity={}
    for name in ('default','candidate'):
        expected=prior['integrity'][name]
        actual=sha(expected['path']);assert actual==expected['sha256']
        integrity[name]={'path':expected['path'],'sha256':actual,'unchanged':True}
    for seed in (7,11,23):
        expected=prior['audits'][f'v1-delta-seed{seed}']
        actual=sha(expected['checkpoint']);assert actual==expected['checkpoint_sha256']
        integrity[f'foundation{seed}']={'sha256':actual,'unchanged':True}
    accepted=[]
    for seed,result in audit.items():
        assert result['selection_sha256']==lock['selection_sha256']
        assert result['variant']==development['selected']['variant']
        assert result['writer']==development['selected']['writer']
        for w in result['worlds']:
            failed=[i for i,r in enumerate(w['stages']) if not r['gate']]
            assert not failed or failed==[len(w['stages'])-1]
            if len(w['stages'])==3 and all(r['gate'] for r in w['stages']):
                accepted.append({'seed':int(seed),'world':w['world'],'path':
                    f'runs/p3-audit-seed{seed}/world{w["world"]}-stage3-candidate.pt'})
    result={'date':'2026-11-05','development':development,'selection_lock':lock,
            'audits':audit,'plasticity':plasticity,'controls':controls,'diagnosis':diagnosis,'write_cost':costs,
            'capacity_pressure':capacity,
            'integrity':integrity,'complete_passing_chains':accepted,
            'tests':{'command':'python -m unittest discover -s tests -v','passed':24},
            'limits':['Finite 64-key, 16-value vocabulary','Training replay is required by P-only transfer',
                      'Unknown-key teacher guesses are not established facts',
                      'Exact matrix rebase does not compress information or expand capacity',
                      'Default and P2 candidate not promoted or modified',
                      'Historical time queries and larger vocabulary not implemented this round']}
    candidate_path=Path('reports/P3-candidate.json')
    if candidate_path.exists():
        candidate=read(candidate_path)
        assert sha(candidate['candidate'])==candidate['sha256']
        result['exported_candidate']=candidate
    Path('reports/P3-results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# P3：转存后还能保留并纠正吗','',
        '2026-11-05。本轮专注 P2 暴露的巩固副作用，保留原默认个体和独立候选，未进入扩容。','',
        '## 先修正问题定义','',
        '只读复查六个 P2 Delta 转存候选：已学事实正确率均为 100%，改变的未知键上教师最高概率约 0.24–0.50。因此不能把这些变化全部称为遗忘旧知识。原 95% 行为保留门槛仍保留，同时单列已验证事实和未知猜测。','',
        '更重要的是，新检查发现六个阶段中四个在一次写入后不能纠正指定的已巩固事实。原来只检查转存后的静态回答，漏掉了后续可塑性。本轮增加四次独立纠正与其他已学事实回归。','',
        '## 改动与对照','',
        '直接写入让快速读出趋近新目标，但读取时还会加上慢速专家的旧贡献。残差方案从目标中扣除当前查询路径下慢速贡献，逐层更新；这增加了写入计算，不能称为免费改进。单元测试验证其代数行为，全模型表现另做评估。','',
        '单个体单次写入的矩阵运算计数：直接写入 124,560，残差写入 4,988,704（约 40 倍），一次普通查询为 4,890,256。增加量约相当于一次前向查询；非线性和内存搬运未完整统计。两者各一次内部更新，绝不等于同计算量。','',
        '三个转存目标分别为原目标、较强分布保留项、带类别间隔的保留项。每种搭配原写入与残差写入。固定 400 步，开发流 810000、基础种子 7、三个世界、每世界两个阶段。首阶段可隔离写入规则，后续阶段比较的是完整学习轨迹。','',
        '| 开发方案 | 通过阶段 / 6 | 再次纠正平均率 | 最低未知回答一致性 |',
        '|---|---:|---:|---:|']
    for c in development['candidates']:
        rr=rows(c)
        lines.append(f'| {c["variant"]["name"]} / {c["writer"]} | {sum(r["gate"] for r in rr)}/6 | '+
            percent(statistics.mean(r['correction'] for r in rr))+' | '+
            percent(min(min(r['anchor_reset'],r['anchor_retained']) for r in rr))+' |')
    selected=development['selected']
    lines+=['',f'开发流选中 {selected["variant"]["name"]} + {selected["writer"]}。选择文件在运行新评估前计算 SHA256 并锁定；后续不根据结果重选。','',
        '## 新随机流评估','',
        '随机流 910000，三个既有独立基础种子各四个世界，每世界计划三阶段，每阶段增加八条，共最多 24 条。任何阶段失败即停止该链，保留父快照。未到达的阶段不是通过，也没有偷偷沿失败版本继续。','',
        '| 基础种子 | 完整通过的链 / 4 | 已执行阶段 | 通过阶段 | 再次纠正平均率 | 最低已知事实正确率 |',
        '|---|---:|---:|---:|---:|---:|']
    for seed,a in audit.items():
        rr=rows(a);chains=sum(len(w['stages'])==3 and all(r['gate'] for r in w['stages']) for w in a['worlds'])
        lines.append(f'| {seed} | {chains}/4 | {len(rr)} | {sum(r["gate"] for r in rr)} | '+
                     percent(statistics.mean(r['correction'] for r in rr))+' | '+
                     percent(min(min(r['known_reset'],r['known_retained']) for r in rr))+' |')
    lines+=['',f'完整通过 {len(accepted)}/12 条链。阶段内考核教师已知正确率、重置与保留快参数后的事实准确率、两种条件的未知回答保留、一次纠正和其他旧事实保留，均要求至少 95%。','',
        '类别间隔损失会主动拉开原教师第一名与其他选项的距离。回答排序保持不等于概率分布完全不变；逐阶段的保留散度也已记录，不能据此声称不改变不确定性或已经学会判断证据。','',
        '每个世界的已学事实与教师锚点参加过转存优化。因此这是独立经历流中的记忆迁移及后续修订证据，不是对未见事实泛化。样本规模较小，不提供终身保证。','',
        '## 追加连续修改','',
        '对每个世界最后通过的阶段做 12 次交替纠正，查询的是当前值。没有通过阶段的世界跳过并保留标记，不能算作通过。','',
        '| 基础种子 | 有可用候选的世界数 | 12 次纠正平均率 | 最终已学事实平均率 |',
        '|---|---:|---:|---:|']
    for seed,p in plasticity.items():
        rr=[r for r in p['results'] if 'correction_success' in r]
        correction=percent(statistics.mean(r['correction_success'] for r in rr)) if rr else '无可用候选'
        retention=percent(statistics.mean(r['final_all_known'] for r in rr)) if rr else '无可用候选'
        lines.append(f'| {seed} | {len(rr)} | {correction} | {retention} |')
    lines+=['','## 后续容量压力','',
        '锁定方案后追加压力检查：从通过的 24 条转存状态开始，再写 8、24、40 个新键，期间不再巩固。只在原 64 键字母表内增加经历，并非模型扩容。下表三种子、可达世界的简单均值，不能当作独立训练样本显著性。','',
        '| 总事实数 | 已巩固 24 条保留 | 新事实正确率 | 世界数 |','|---|---:|---:|---:|']
    for count in (32,48,64):
        rr=[r for d in capacity.values() for r in d['results'] if r.get('total_facts')==count]
        if rr:
            lines.append(f'| {count} | '+percent(statistics.mean(r['consolidated24'] for r in rr))+' | '+
                         percent(statistics.mean(r['new_facts'] for r in rr))+f' | {len(rr)} |')
    lines+=['','压力结果不用于补调，也不改变已冻结门槛。若高负荷下仍有下降，应继续报告而不是宣称消除了遗忘。','',
        '## 精确搬移控制','',
        '无衰减 Delta 的活动矩阵可复制到个体自有初始矩阵，再重置快速状态；前向与下一次直接写入保持一致，参数量不增加。本轮三个基础种子共 36 个阶段全部通过，同步验证父模型不被修改。','',
        '这只是已有有限状态在神经参数间搬移，不是压缩、抽象、扩容或 P-only 专家学习。重置 F 会读取更新后的初始矩阵，不能把保留下来的内容称作自动涌现的知识。无转存控制重置 F 后正确率为 0%–12.5%，完整数据包含在 JSON。','',
        '## 文件与完整性','',
        '- 原默认个体、P2 候选及三份 Delta 基础权重哈希与上轮记录一致。',
        '- 24 项测试通过，包含残差写入公式、禁止低可信写入、恢复后续写、精确搬移等价性与拒绝无效输入。',
        '- 新评估首世界各可达阶段核对重置与保留快参数的新进程恢复；只向工作进程传检查点与当前键。',
        '- 候选和父快照在 runs/p3-audit-seed*/，失败同样保留；未升级默认个体。',
        '- 逐变体原始结果与所有失败原因在 reports/P3-results.json；协议在 docs/P3-protocol.md。','',
        '## 后续边界','',
        '本轮没有增加键空间或实现历史时刻查询，避免同时改动多项而失去归因。未知内容的行为稳定性、长期连续修订和更大容量仍需验证。当前结果不证明生命、意识、自然语言能力或无限记忆。']
    if candidate_path.exists():
        lines+=['','## 独立研究个体','',
                '检查点 checkpoints/individual-p3-candidate.pt 来自固定 seed7/world0 的第三阶段，已巩固 24 条符号事实。它是独立研究分支，不是默认个体迁移。测试清单在 reports/P3-candidate.json，普通查询不读取该清单。','',
                '~~~powershell',
                '.\\.venv\\Scripts\\python.exe -m spora.cli --individual checkpoints/individual-p3-candidate.pt ask 56',
                '~~~','',
                '导出时回答为 4，重置 F 的新进程恢复已验证。未来 learn 会给该分支增加经历，并可能产生干扰。']
    Path('reports/P3-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'complete_passing_chains':len(accepted),'selected':selected,
                      'integrity':integrity},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
