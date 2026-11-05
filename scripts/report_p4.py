"""Assemble measured P4 evidence; failed gates cannot create a release individual."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spora.checkpoint import load_checkpoint


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pct(value):return '不适用' if value is None else f'{100*value:.2f}%'


def aggregate(rows):
    names=rows[0]
    return {name:{'correct':sum(row[name]['correct'] for row in rows),
                  'total':sum(row[name]['total'] for row in rows),
                  'accuracy':sum(row[name]['correct'] for row in rows)/sum(row[name]['total'] for row in rows)
                      if sum(row[name]['total'] for row in rows) else None,
                  'seed_min':min([row[name]['accuracy'] for row in rows if row[name]['accuracy'] is not None],default=None),
                  'seed_max':max([row[name]['accuracy'] for row in rows if row[name]['accuracy'] is not None],default=None)}
            for name in names}


def cpu_timing(path):
    model,fast,velocity,_=load_checkpoint(path)
    key=torch.tensor([12]);value=torch.tensor([5]);stamp=torch.tensor([0]);trust=torch.ones(1)
    with torch.no_grad():
        for _ in range(5):model(key,fast)
        start=time.perf_counter()
        for _ in range(50):model(key,fast)
        query=(time.perf_counter()-start)/50
        start=time.perf_counter()
        for _ in range(50):model.write_temporal(fast,velocity,key,value,stamp,trust)
        write=(time.perf_counter()-start)/50
    return {'cpu_threads':2,'repeats':50,'current_query_ms':query*1000,'dual_bank_event_ms':write*1000,
            'note':'CPU microbenchmark after warmup, excludes checkpoint I/O; not an equal-compute comparison'}


def main():
    torch.set_num_threads(2)
    output=ROOT/'reports'/'P4-results.json';markdown=ROOT/'reports'/'P4-report.md'
    if output.exists() or markdown.exists():raise FileExistsError('Preserve previous report')
    old={'individual.pt':'85b34a36ab567586740e895411d3569facac9312cdeb4504d1f168f4098be037',
         'individual-p2-candidate.pt':'2c94e6de0cc20054407828a158396804115b872fffd4edb81dda6726bd4e6961',
         'individual-p3-candidate.pt':'9709fd47df9d17d173dbdb2bd4536966361791b2ec1ad24c8dc997eb547edf1a'}
    integrity={name:{'expected':sha,'actual':digest(ROOT/'checkpoints'/name)} for name,sha in old.items()}
    if any(row['expected']!=row['actual'] for row in integrity.values()):raise AssertionError('Original individual changed')
    test_text=(ROOT/'runs'/'p4-final-tests.log').read_text(encoding='utf-8')
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',test_text)
    if not match or not test_text.rstrip().endswith('OK'):raise AssertionError('Completed regression run required')
    regression={'tests':int(match.group(1)),'seconds':float(match.group(2)),'passed':True}
    models=[];summary={}
    for kind in ('split','shared'):
        group=[]
        for seed in (7,11,23):
            name=f'p4-{kind}-seed{seed}';run=ROOT/'runs'/name
            training=read(run/'summary.json');config=read(run/'config.json')
            audit=read(ROOT/'runs'/f'{name}-audit'/'audit.json')
            if digest(Path(audit['checkpoint']))!=audit['sha256']:raise AssertionError('Audited foundation changed')
            if training['outer_steps']!=380 or training['event_presentations']!=19840:raise AssertionError('Budget mismatch')
            row={'name':name,'kind':kind,'seed':seed,'configuration':config,'training':training,
                 'audit':audit,'timing':cpu_timing(Path(audit['checkpoint']))}
            models.append(row);group.append(audit)
        summary[kind]={'main':{str(n):aggregate([a['main'][str(n)] for a in group]) for n in (8,16,32)},
                       'pressure64':aggregate([a['pressure64'] for a in group]),
                       'controls':{name:aggregate([a['controls'][name] for a in group]) for name in group[0]['controls']},
                       'immediate':aggregate([a['immediate'] for a in group]),
                       'early_diagnostic':aggregate([a['early_diagnostic'] for a in group]),
                       'unseen_pairs_diagnostic':aggregate([a['unseen_pairs_diagnostic'] for a in group]),
                       'passed_seeds':[a['metadata']['seed'] for a in group if a['gate']['passed']]}
    fixed=models[0]['audit']
    if fixed['gate']['passed']:
        raise RuntimeError('Fixed seed7 passed: perform explicit candidate export/restart before finalizing report')
    result={'date':'2026-11-05','parent_commit':'1955d01','plan':read(ROOT/'configs'/'p4-plan.json'),
            'summary':summary,'models':models,'original_individual_integrity':integrity,
            'fixed_demo':read(ROOT/'runs'/'p4-fixed-demo'/'result.json'),
            'restart_standardized':read(ROOT/'runs'/'p4-restart-standardized.json'),
            'regression':regression,
            'candidate':{'exported':False,'reason':'Fixed seed7 split fails preregistered gate',
                         'failed_metrics':fixed['gate']['failures']},
            'scope':'Independent random initialization, finite symbolic observation history; no P3 migration or consciousness claim'}
    lines=['# P4：当前判断与历史经历的分离实验','',
           '2026-11-05。完成两种结构、三个独立训练种子的固定预算实验。新架构均从随机权重开始；原有 P1、P2、P3 个体保持原文件。',
           '', '**结论：分区实现了当前修订与过往观察的分离，但较长历史仍存在明显遗忘。固定 seed7 的 split 未通过全部 95% 门槛，本轮不导出发布候选、不替换默认个体。**','',
           '## 主审计','',
           '每种结构每个种子、每个长度评估 64 个世界；下表合计 192 次世界运行。三个种子复用同一组环境样本，不能当作 192 个独立环境。每个世界只有四个实体反复变化，不能外推为相同数量的独立事实容量。所有成绩来自最后课程完成后的权重。','',
           '| 结构 | 事件数 | 当前值 | 有效历史 | 历史前半段 | 错误组合拒答 | 未来拒答 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for kind in ('split','shared'):
        for n in (8,16,32):
            row=summary[kind]['main'][str(n)]
            lines.append('| '+kind+' | '+str(n)+' | '+' | '.join(pct(row[m]['accuracy']) for m in
                ('current','history','old_history','invalid_pair_unknown','future_unknown'))+' |')
    lines+=['','分区在本次样本中提高了历史回忆，但 32 事件时对错误组合与未来的拒答弱于共享方案；不是所有指标都占优。16 事件的均值超过门槛，也不能替代逐种子逐指标检查。','',
            '按种子列出 32 事件历史成绩，避免均值遮住差异：','',
            '| 结构 | seed7 | seed11 | seed23 | 全长度门槛通过种子 |','|---|---:|---:|---:|---|']
    for kind in ('split','shared'):
        rows=[m for m in models if m['kind']==kind]
        lines.append('| '+kind+' | '+' | '.join(pct(m['audit']['main']['32']['history']['accuracy']) for m in rows)
                     +' | '+(str(summary[kind]['passed_seeds']) if summary[kind]['passed_seeds'] else '无')+' |')
    lines+=['','## 干预与恢复','',
            '以下对照统一使用 16 事件。重置历史区是恢复初始神经矩阵；没有从文件重放原始事件。时间置乱仍可能碰巧落到相同值，因此不把零准确率当作必须条件。','',
            '| 结构 | 对照 | 当前值 | 有效历史 | 错误组合拒答 |','|---|---|---:|---:|---:|']
    for kind in ('split','shared'):
        for name,row in summary[kind]['controls'].items():
            lines.append('| '+kind+' | '+name+' | '+' | '.join(pct(row[m]['accuracy']) for m in
                ('current','history','invalid_pair_unknown'))+' |')
    exact=sum(m['audit']['restart']['identical_probabilities'] and m['audit']['restart']['continuation_exact_tensors'] for m in models)
    standardized=result['restart_standardized']['models'];verified=sum(r['same_threads_exact'] and r['file_unchanged'] for r in standardized)
    largest=max(r['threads4_vs2_max_probability_difference'] for r in standardized)
    if not all(r['threads4_vs2_prediction_match'] for r in standardized):raise AssertionError('Unexpected thread prediction difference')
    lines+=['',f'原审计 {exact}/6 通过概率逐项一致及继续写入参数逐项一致。第一组进程在统一线程设置修补前已启动，保留其不同线程比较失败记录；随后对六份原状态追加两线程对两线程复核，{verified}/6 概率逐项一致且查询前后文件校验值不变。四线程与两线程最大概率差 {largest:.3g}，各组类别预测一致。补充记录独立保存，不重写主审计，不改变容量失败结论。',
            '', '新进程工作目录为空，模型加载后由 Python 审计钩子禁止文件、目录、进程及网络操作；这不是操作系统沙箱。',
            '', '固定演示依次接收 12→5（可信）、12→9（可信）、12→3（不可信）。查询顺序为当前、t0、t1、t2、未来 t3、错误键13/t0；unknown 用类别 16 表示。',
            '', '预期：[9, 5, 9, 3, 16, 16]；实际：'+str(result['fixed_demo']['prediction'])+'。清空活动记忆后：'+str(result['fixed_demo']['reset_prediction'])+'。仅为固定示例，不代替总体审计，状态保留在 runs/p4-fixed-demo/。',
            '', '当前语义为最后一次可信赋值；历史语义为某时刻实际听到的值，包含不可信观察，不等于当时的世界真相。可信度为环境提供的二元标签，没有训练自主辨伪。','',
            '| 结构 | 一步可信纠正 | 不可信后当前值 | 不可信前后预测一致 | 三次早期观察历史 |',
            '|---|---:|---:|---:|---:|']
    for kind in ('split','shared'):
        row=summary[kind]['immediate'];early=summary[kind]['early_diagnostic']
        lines.append('| '+kind+' | '+' | '.join(pct(row[m]['accuracy']) for m in
            ('correction','untrusted_current','untrusted_preserves_prediction'))+' | '+pct(early['all_three_history']['accuracy'])+' |')
    lines+=['','早期诊断先接收不可信值，再连续两次可信修订，属于不同于训练前四条必可信的分布，结果单列、不参与挑选。完整计数、unknown 与可信/不可信历史分组见 P4-results.json。',
            '', '| 结构 | 其他偏移错误键拒答 | 未观察的过去拒答 |', '|---|---:|---:|']
    for kind in ('split','shared'):
        row=summary[kind]['unseen_pairs_diagnostic']
        lines.append('| '+kind+' | '+' | '.join(pct(row[m]['accuracy']) for m in
            ('distant_wrong_key_unknown','unobserved_past_unknown'))+' |')
    lines+=['', '## 历史位置差异','',
            '32 事件按发生先后分为四段，每段八次。下面每格都是同一最终状态的读取结果。','',
            '| 结构 | 最早八次 | 第 9–16 次 | 第 17–24 次 | 最近八次 |','|---|---:|---:|---:|---:|']
    for kind in ('split','shared'):
        row=summary[kind]['main']['32']
        lines.append('| '+kind+' | '+' | '.join(pct(row[f'history_quarter_{q}']['accuracy']) for q in range(1,5))+' |')
    lines+=[
            '', '## 容量压力与成本','',
            '| 结构 | 64 事件当前值 | 有效历史 | 历史前半段 | 错误组合拒答 |','|---|---:|---:|---:|---:|']
    for kind in ('split','shared'):
        row=summary[kind]['pressure64']
        lines.append('| '+kind+' | '+' | '.join(pct(row[m]['accuracy']) for m in
            ('current','history','old_history','invalid_pair_unknown'))+' |')
    lines+=['','64 事件超过最长训练流；未来时刻没有合法词表位置，因此该项为 null。达到时间 64 时交互入口明确拒绝保存，不循环覆盖。','',
            '| 结构 | 主模型参数 | 活动快参数 | 本结构三次训练时间范围/秒 | 峰值已分配显存最大/MiB |',
            '|---|---:|---:|---:|---:|']
    for kind in ('split','shared'):
        rows=[m for m in models if m['kind']==kind];seconds=[m['training']['elapsed_seconds'] for m in rows]
        lines.append(f"| {kind} | {rows[0]['configuration']['parameters']:,} | {rows[0]['configuration']['fast_parameters']:,} | {min(seconds):.1f}–{max(seconds):.1f} | {max(m['training']['peak_cuda_allocated_bytes'] for m in rows)/2**20:.1f} |")
    lines+=['','每个模型 380 次外层更新、19,840 次事件呈现；六模型共 2,280 次更新、119,040 次呈现。两种结构只近似匹配活动快参数，未匹配总参数或总运算量。训练耗时受同机任务影响；JSON 提供 CPU 两线程查询及写入微测量，不据此宣称等算力优势。',
            '', '## 交付与边界','',
            f"- 回归测试 {regression['tests']} 项全部通过；原有三个个体 SHA256 均与实验前一致。",
            '- 六个研究底座位于 checkpoints/p4-{split,shared}-seed{7,11,23}.pt，包含训练状态，不是已发布个体。',
            '- 新入口 spora.temporal_cli 支持显式 create、learn、ask 与 --time；必须指定个体路径。旧入口拒绝对时间结构操作，防止丢失历史区。',
            '- 每次外部观察只推进一次计数；保存采用临时文件替换及单写者锁。越界、非法值和已有锁不会覆盖原文件。',
            '- 测评用恢复状态在 runs/，它们不是候选发布。模型文件只保存张量、配置和非答案元信息，不保存事件数组或外部查询表。',
            '- 两区及写入门控是人为指定的结构先验；不代表已经涌现自主身份、来源判断或生命意识。',
            '- 本轮未整合 P3 慢速巩固，不是旧个体的连续迁移；活动快矩阵是可持久神经参数，但骨干在在线经历阶段不更新。',
            '- 时间词表固定 64，键词表固定 64，输出值 16；不是自然语言理解、无限记忆或时间范围外泛化。',
            '', '## 下一项优先实验','',
            '先改善同一历史区的地址干扰，再增加长期事件量：固定当前分区，比较更低相关性的时序地址、有限多个可学习历史子区，以及同活动参数预算的单矩阵基线。将早期事件与错误组合拒答一起约束，不能仅提高总回忆率而增加虚构。重新预登记开发流及审计流，保留本轮失败基线。仅扩大矩阵维度可以作为容量对照，不应冒充机制进步。',
            '', '依据和推导见 docs/P4-protocol.md；参考研究只支持局部机制，最终结论来自上述实测。']
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    markdown.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(markdown),'candidate':result['candidate'],'integrity':'all original hashes match'},ensure_ascii=False))


if __name__=='__main__':main()
