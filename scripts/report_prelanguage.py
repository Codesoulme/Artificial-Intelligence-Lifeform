"""Gather all completed iterations with strict prerequisite checks."""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def percent(x):return '不适用' if x is None else f'{100*x:.2f}%'

def main():
    dest=ROOT/'reports/prelanguage-results.json';doc=ROOT/'reports/prelanguage-report.md'
    if dest.exists() or doc.exists():raise FileExistsError('Retain prior published reports')
    original={'individual.pt':'85b34a36ab567586740e895411d3569facac9312cdeb4504d1f168f4098be037',
              'individual-p2-candidate.pt':'2c94e6de0cc20054407828a158396804115b872fffd4edb81dda6726bd4e6961',
              'individual-p3-candidate.pt':'9709fd47df9d17d173dbdb2bd4536966361791b2ec1ad24c8dc997eb547edf1a'}
    integrity={name:hashlib.sha256((ROOT/'checkpoints'/name).read_bytes()).hexdigest() for name in original}
    if integrity!=original:raise AssertionError('Old individual changed')
    foundations={};audits={};growth={};negatives={}
    for variant,seeds,prefix in [('rls90',(7,11,23),'p5'),('delta110',(7,),'p5'),('orthogonal90',(7,11,23),'p5b')]:
        for seed in seeds:
            name=f'{prefix}-{variant}-seed{seed}'
            foundations[name]={'configuration':read(f'runs/{name}/config.json'),'training':read(f'runs/{name}/summary.json')}
            audit=ROOT/f'runs/{name}-audit/audit.json'
            if audit.exists():audits[name]=json.loads(audit.read_text())
            if variant=='rls90' and not audit.exists():raise AssertionError('Original iteration audit incomplete')
    for seed in (7,11,23):
        negatives[str(seed)]=read(f'runs/p6-negatives-seed{seed}.json')
        if not negatives[str(seed)]['passed']:raise AssertionError('Temporal absence gate failed')
        if not audits[f'p5b-orthogonal90-seed{seed}']['gate']['passed']:raise AssertionError('Memory gate failed')
        for entities in (4,64):
            name=f'p6-seed{seed}-entities{entities}';growth[name]=read(f'runs/{name}/audit.json')
            if not growth[name]['passed']:raise AssertionError('Growth gate failed')
    readiness=read('runs/p7-preparation/readiness.json')
    if not readiness['passed'] or readiness['Chinese_training_started'] or readiness['trained_language_tokens']!=0:raise AssertionError('Readiness not met')
    test=(ROOT/'runs/prelanguage-final-tests.log').read_text(encoding='utf-8');match=re.search(r'Ran (\d+) tests in ([\d.]+)s',test)
    if not match or not test.rstrip().endswith('OK'):raise AssertionError('Final tests required')
    result={'date':'2026-11-05','starting_commit':'95c2956','plans':{n:read(f'configs/{n}.json') for n in ('p5-plan','p5b-plan','p6-plan')},
            'foundations':foundations,'audits':audits,'growth':growth,'temporal_absence':negatives,'old_hashes':integrity,
            'p5_varied_development_failure':read('runs/p5-varied-development.json'),
            'candidate':read('runs/prelanguage-candidate.json'),'language_readiness':readiness,
            'regression':{'count':int(match.group(1)),'seconds':float(match.group(2)),'passed':True},
            'total_training':{'models':len(foundations),'outer_steps':sum(x['training']['steps'] for x in foundations.values()),
                              'event_presentations':sum(x['training']['event_presentations'] for x in foundations.values())},
            'ready_for_bounded_chinese_experiment':True,'Chinese_training_started':False}
    lines=['# 中文训练前：连续迭代交付','',
           '2026-11-05。从 P4 的历史遗忘继续开发，完成 P5 相关性保护、P5b 多实体修订、P6 同个体分段成长，以及 P7 未训练字节接口。当前达到进入小规模中文试验的工程门槛，实际中文训练尚未开始。','',
           '**不等于具备中文能力、无限记忆、开放世界实体理解或生命意识。** 所有底座都来自随机初始化；既有个体原文件不变。新谱系和旧谱系没有被混称为同一个体。','',
           '## 迭代中发现什么','',
           '1. P4 在 64 事件时历史回忆为 60.21%。P5 改用递归最小二乘保护历史地址，同时从逆相关矩阵学习未知判别。','2. 同持久状态规模的 Delta 对照仍有遗忘；并非仅增大矩阵就达到新结果。但 RLS 与拒答头同时变化，本轮不能把收益全部归因于单独一个因素。',
           '3. P5 的额外开发检查发现：64 个不同实体时当前值只有 91.43%，未知拒答只有 7.28%。这项失败保留，未用四实体成绩掩盖。',
           '4. P5b 增加有限符号的随机正交地址与当前读出能量判别，混合实体分布训练；换用新的审计流复核。正交地址为显式设计先验，并非模型自行发现身份。','',
           '## 独立审计：最终修订','',
           '每个模型使用新流的 64 世界，三个种子各自验收。相同环境样本跨种子复用，不应把重复运行数当作独立环境数。每个门槛按种子和长度逐项检查，未凭汇总均值发布。','',
           '| 种子 | 事件数 | 当前值 | 历史 | 早期历史 | 错误组合拒答 | 未来拒答 |','|---|---:|---:|---:|---:|---:|---:|']
    for seed in (7,11,23):
        for n in (8,16,32,64):
            row=audits[f'p5b-orthogonal90-seed{seed}']['main'][str(n)]
            lines.append('| '+str(seed)+' | '+str(n)+' | '+' | '.join(percent(row[k]['accuracy']) for k in ('current','history','old_history','invalid_pair_unknown','future_unknown'))+' |')
    lines+=['','64 事件的最后时刻没有词表内未来位置，因此该项为不适用；其余长度未来拒答仍须达标。P4 只训练到 32 事件，本轮训练到 64，预算也从每模型 380 步增至 600；P4→P5 的差异不能单独归因于算法。P5 内部的 Delta/RLS 对照才使用相同步数和事件长度。','',
            '| 种子 | 64 不同实体当前值 | 历史 | 未可信当前拒答 | 错误组合拒答 |','|---|---:|---:|---:|---:|']
    for seed in (7,11,23):
        row=audits[f'p5b-orthogonal90-seed{seed}']['varied_entities64']
        lines.append('| '+str(seed)+' | '+' | '.join(percent(row[k]['accuracy']) for k in ('current','history','current_unknown','invalid'))+' |')
    lines+=['','| 种子 | 同键错误时刻拒答 | 已分配但未经历时间拒答 |','|---|---:|---:|']
    for seed in (7,11,23):
        row=negatives[str(seed)];lines.append('| '+str(seed)+' | '+percent(row['same_key_wrong_time']['accuracy'])+' | '+percent(row['empty_allocated_pages']['accuracy'])+' |')
    lines+=['','## 同个体跨段成长','',
            '每个种子分别运行四实体与 64 实体条件，各八个世界；每个世界连续 256 次观察，64/128/192/256 时保存恢复。扩展不改已有张量与旧查询输出，只增添神经矩阵，成本随容量增加。每个世界每个阶段均须过门槛。','',
            '| 种子 | 实体数 | 最低当前值 | 最低历史 | 最低最早64次 | 最低错误组合拒答 | 新进程一致 |','|---|---:|---:|---:|---:|---:|---|']
    for seed in (7,11,23):
        for entities in (4,64):
            g=growth[f'p6-seed{seed}-entities{entities}'];stages=[s for w in g['worlds'] for s in w['stages']]
            lines.append('| '+str(seed)+' | '+str(entities)+' | '+' | '.join(percent(min(s[k] for s in stages)) for k in ('current','history','oldest64','invalid_unknown'))+' | '+str(sum(w['cold_exact'] for w in g['worlds']))+'/8 |')
    lines+=['','分段按明确的时间区间路由，是工程先验。它不是文本检索，也不是无成本压缩；这里只证明在已分配有限范围内能连续保存和读取。P5 个体采用其真实矩阵迁入增长格式，不从原事件重放；P1–P4 跨结构迁移仍未解决。','',
            '## 参数来源与消融','',
            '| 种子 | 正常64历史 | 不写入历史 | 重置历史区后 | 重置当前区后的历史 | 时间置乱历史 |','|---|---:|---:|---:|---:|---:|']
    for seed in (7,11,23):
        a=audits[f'p5b-orthogonal90-seed{seed}'];c=a['controls']
        lines.append('| '+str(seed)+' | '+percent(a['main']['64']['history']['accuracy'])+' | '+' | '.join(percent(c[k]['history']['accuracy']) for k in ('no-write','reset-history','reset-current','shuffled-time'))+' |')
    lines+=['','查询输入只有键、模式、时刻；恢复测试从空工作目录启动，加载后禁止文件与网络访问。Python 审计钩子不是操作系统沙箱。逆相关矩阵含地址几何证据，属于必须计费的内部优化器状态，不能省略。','',
            '## 资源与边界','',
            '| 模型 | 外层步数 | 事件呈现 | 训练秒数 | 峰值已分配显存 MiB |','|---|---:|---:|---:|---:|']
    for name,data in foundations.items():
        t=data['training'];lines.append(f"| {name} | {t['steps']} | {t['event_presentations']} | {t['seconds']:.1f} | {t['peak_cuda_bytes']/2**20:.1f} |")
    candidate=result['candidate']
    lines+=['',f"最终个体模型参数 {candidate['trainable_and_frozen_model_parameters']:,}，固定地址缓冲区 {candidate['basis_buffer_scalars']:,}；四段活动状态（含逆相关证据）{candidate['state_scalars']['total']:,} 标量。单段为 48,600，四段为 145,800。重复训练曾并行运行，因此时间只报告实测，不用于公平速度排名。",
            '', '## 中文训练前接口','',
            f"新增接口参数 {readiness['interface_parameters']:,}；UTF-8 字节词表共 259 类。随机字节检查通过因果前缀、梯度、接口恢复及原个体不变，真实语言训练 token 数为零。保存的接口仍是随机初值。",
            '', '复用个体因果骨干并只读访问神经记忆；第一阶段只训练新接口。其尚未学会中文语义与实体/关系/时间对齐，不保证自由文本能准确表达内部回忆。冻结保护既有符号能力，但可能限制语言学习；后续是否解冻要依据新的实验。',
            '', '训练入口 spora.train_language 要求显式启动，来源清单必须含三个文档级分区、摘要与许可。当前仅实现空白规范化后的完全去重，不等于真实语料的近似去重与隐私审查已完成。本轮没有下载中文语料。',
            '', '## 本地交付','',
            '- 候选：checkpoints/individual-prelanguage.pt，固定 seed7，已有三次演示经历、四段分配容量；默认 P1 个体不变。',
            '- 未训练接口：runs/p7-preparation/untrained-interface.pt。',
            '- 查询示例：当前12→9；t0→5、t1→9、t2→3；不可信观察保留为历史，不改变当前判断。',
            f"- {result['regression']['count']} 项回归通过；原有三份个体文件 SHA256 与起点相同。",
            '- 完整数字与失败版本见 prelanguage-results.json；权重与本地上下文不进入 GitHub。',
            '- P3 慢速巩固还未与新架构合并；开放实体、来源可信度学习、语言对齐、行动和意识均未完成。',
            '', '## 停止点','',
            '已达到启动有界中文试验的工程准备条件，停在实际语料训练之前。下一步先确定来源清楚的小规模中文语料及预算，再评估语言损失、语义对齐和旧能力保持；不把本报告当作已经形成中文能力的证明。']
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(doc),'ready':True,'Chinese_training_started':False},ensure_ascii=False))

if __name__=='__main__':main()
