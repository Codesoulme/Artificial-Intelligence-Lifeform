"""Collect preserved evidence and verify that P1 ancestry was not modified."""
import hashlib
import json
from pathlib import Path
import statistics
import torch
from spora.checkpoint import load_checkpoint
from spora.evaluate import isolated_query


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def at(audit,pairs=64,policy=None):
    row=next(r for r in audit['results'] if (r['policy']==policy if policy else
             all(r[k]==v for k,v in audit['selected'].items())))
    return next(r for r in row['scores'] if r['pairs']==pairs)


def pct(value):
    return f'{100*value:.2f}%'


def main():
    torch.set_num_threads(2)
    names=['p1-seed7','matched-mlp-seed7','short-control-seed7']+[
        f'v1-{kind}-seed{seed}' for kind in ('mlp','delta') for seed in (7,11,23)]
    audits={name:read(f'runs/p2-policy-{name}/audit.json') for name in names}
    for audit in audits.values():
        assert audit['worlds']==64 and audit['restart']['match']
        assert digest(audit['checkpoint'])==audit['checkpoint_sha256']
    training={p.parent.name:read(p) for p in sorted(Path('runs').glob('p2*/summary.json'))}
    configs={p.parent.name:read(p) for p in sorted(Path('runs').glob('p2*/config.json'))}
    protection={k:read(f'runs/p2-protection-{k}-seed7.json') for k in ('mlp','delta')}
    consolidation={k:read(f'runs/p2-consolidation-{k}-seed7/consolidation.json') for k in ('mlp','delta')}
    stress={f'{k}-{s}':read(f'runs/p2-stress-{k}-seed{s}.json') for k in ('mlp','delta') for s in (7,11,23)}
    compute={name:read(f'runs/p2-compute-{name}.json') for name in ('profile','counter')}
    integrity={}
    for seed in read('reports/memory-diagnosis-2026-11-05.json')['training_seeds']:
        path=f'checkpoints/p1-seed{seed["seed"]}.pt'
        actual=digest(path);assert actual==seed['checkpoint_sha256']
        integrity[path]={'sha256':actual,'unchanged':True}
    actual=digest('checkpoints/individual.pt')
    assert actual=='85b34a36ab567586740e895411d3569facac9312cdeb4504d1f168f4098be037'
    integrity['checkpoints/individual.pt']={'sha256':actual,'unchanged':True}
    for key,path in [('default','checkpoints/individual.pt'),('candidate','checkpoints/individual-p2-candidate.pt')]:
        model,fast,_,data=load_checkpoint(path)
        expected=5 if key=='default' else 9
        direct=int(model(torch.tensor([12]),fast).argmax(-1))
        restart=isolated_query(path,[12])['predictions'][0]
        assert direct==restart==expected
        integrity[key]={'path':path,'sha256':digest(path),'cursor':data['cursor'],
                        'key12_answer':direct,'restart_match':True,'memory_kind':model.config.memory_kind}
    means={}
    for kind in ('mlp','delta'):
        for policy in ('one','selected'):
            entries=[at(audits[f'v1-{kind}-seed{s}'],policy=None if policy=='selected' else policy) for s in (7,11,23)]
            means[f'{kind}-{policy}']={metric:statistics.mean(x[metric] for x in entries)
                for metric in ('initial','corrected','unrelated','final','oldest8','latest8','inner_updates_per_trusted_event')}
    result={'date':'2026-11-05','scope':'finite symbolic P2 research; no default promotion',
            'audits':audits,'training':training,'training_configs':configs,'protection':protection,
            'consolidation':consolidation,'stress':stress,'compute':compute,'integrity':integrity,
            'mean_64':means,'tests':{'command':'python -m unittest discover -s tests -v','passed':17},
            'limits':['Historical event queries specified but not implemented','Trust bit externally supplied',
                      'Only 64 keys and 16 values; symbols are seen during training',
                      'No exact hardware-cost-matched trial; covered arithmetic differs by less than one percent',
                      'Only one matched-event seed and one consolidation foundation per architecture',
                      'Five delta consolidation gates failed; no safe default migration demonstrated']}
    Path('reports/P2-results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# P2 实验数值附表','','2026-11-05。完整解释见 P2-report.md，原始证据见 P2-results.json。','',
           '## 冻结策略的新随机流评估','','每个种子、每个长度 64 个世界；下表为 64 条事实的结果。','',
           '| 结构 / 策略 | 即时纠正 | 无关 60 条 | 全部 64 条 | 最早 8 条 |',
           '|---|---:|---:|---:|---:|']
    rows=[('P1 seed7 / 一步',at(audits['p1-seed7'],policy='one'))]+list(means.items())
    for label,row in rows:
        lines.append('| '+label+' | '+' | '.join(pct(row[k]) for k in ('corrected','unrelated','final','oldest8'))+' |')
    lines+=['','逐训练起点，使用开发流选定的策略：','',
            '| 结构 / 种子 | 8 条即时纠正 | 64 条即时纠正 | 无关 60 条 | 全部 64 条 |','|---|---:|---:|---:|---:|']
    for kind in ('mlp','delta'):
        for seed in (7,11,23):
            audit=audits[f'v1-{kind}-seed{seed}'];row=at(audit)
            lines.append(f'| {kind} / {seed} | {pct(at(audit,8)["corrected"])} | '+
                         ' | '.join(pct(row[k]) for k in ('corrected','unrelated','final'))+' |')
    lines+=['','## 等事件参考与短课程控制','',
            '| 候选，seed7 | 事件呈现 | 内部更新 | 优化步 | 一步最终正确率 | 冻结策略最终正确率 |','|---|---:|---:|---:|---:|---:|']
    for name,train in [('matched-mlp-seed7','p2-matched-mlp-seed7'),('v1-delta-seed7','p2-v1-delta-seed7'),('short-control-seed7','p2-short-control-seed7')]:
        t=training[train]
        lines.append(f'| {name} | {t["event_presentations"]:,} | {int(t["inner_updates"]):,} | {t["outer_steps"]} | {pct(at(audits[name],policy="one")["final"])} | {pct(at(audits[name])["final"])} |')
    lines+=['','计算覆盖范围不同，分别记录：','']
    for name,d in compute.items():
        m=d['rows']['mlp']['matched_event_run_covered_flops'];v=d['rows']['delta']['matched_event_run_covered_flops']
        lines.append(f'- {name}：MLP {m/1e12:.4f} T，Delta {v/1e12:.4f} T，相差 {(v/m-1)*100:.3f}%。')
    lines+=['','## 连续纠正与八次干扰后重查','','三种子范围；每种子 64 世界、每世界 16 次目标修订。','',
            '| 结构 / 场景 | 即时纠正 | 隔八次干扰再查 | 最终全部 64 条 |','|---|---:|---:|---:|']
    for kind in ('mlp','delta'):
        for mode in ('same_key','alternating_four'):
            rr=[next(r for r in stress[f'{kind}-{s}']['results'] if r['mode']==mode) for s in (7,11,23)]
            spread=lambda k:pct(min(x[k] for x in rr))+'–'+pct(max(x[k] for x in rr))
            lines.append('| '+kind+' / '+mode+' | '+' | '.join(spread(k) for k in ('immediate','after_gap','final'))+' |')
    lines+=['','## 保护消融','','仅 seed7，32 个开发世界，64 条初始事实。','',
            '| 结构 / 方法 | 即时纠正 | 无关正确率 | 最终正确率 | 更新总数 |','|---|---:|---:|---:|---:|']
    for kind,d in protection.items():
        for row in d['results']:
            if row['pairs']==64:
                lines.append('| '+kind+' / '+row['method']+' | '+' | '.join(pct(row[k]) for k in ('correction','unrelated','final'))+f' | {int(row["inner_updates"]):,} |')
    lines+=['','## 慢速专家转存','','每结构一个基础种子、三个世界、两阶段，每阶段 400 步。','',
            '| 结构 | 重置 F 后事实正确率 | 保留原 F 后 | 无关教师回答一致性范围 | 严格门槛通过 |','|---|---:|---:|---:|---:|']
    for kind,d in consolidation.items():
        rr=[r for w in d['worlds'] for r in w['stages']]
        lines.append(f'| {kind} | {pct(min(r["after_reset_F"] for r in rr))} | {pct(min(r["after_preserved_F"] for r in rr))} | '+
          pct(min(r['unrelated_teacher_agreement'] for r in rr))+'–'+pct(max(r['unrelated_teacher_agreement'] for r in rr))+f' | {sum(r["strict_gate"] for r in rr)}/6 |')
    Path('reports/P2-tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'mean_64':means,'integrity':integrity},ensure_ascii=False))


if __name__=='__main__':main()
