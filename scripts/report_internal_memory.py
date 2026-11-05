"""Collect measured results and explicitly retain rejected development versions."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_internal_runtime import protected


def main():
    protected()
    read=lambda name:json.loads((ROOT/name).read_text(encoding='utf-8'))
    memory=read('runs/p12-internal-audit-v2/audit.json');runtime=read('runs/p12-runtime-audit-v2/audit.json')
    routing=read('runs/p13-read-audit/audit.json');candidate=read('runs/p12-v2-candidate.json');cost=read('runs/p12-cost/profile.json')
    tests=(ROOT/'runs/p12-final-regression.log').read_text(encoding='utf-8');count=re.search(r'Ran (\d+) tests',tests)
    if not all(d['passed'] for d in (memory,runtime,candidate)) or not routing['correctness_passed'] or not count or not tests.rstrip().endswith('OK'):
        raise RuntimeError('Required gates incomplete')
    failed_source=read('runs/p12-internal-audit/audit.json')
    report={'date':'2026-11-05','parent_commit':'646e49d4294a2eb85415904f27039946a41f4b57',
        'write_and_provenance':memory,'runtime':runtime,'routing':routing,'cost':cost,'candidate':candidate,
        'rejected_development':[read(f'runs/p12-seed{s}/summary.json') for s in (7,11)],
        'rejected_continuous_provenance':{'original_audit':failed_source,
          'publication_accepted':False,'reason':'Additional numeric field audit found up to 15 steps of effective-time regression error; replaced by bit-coded finite outputs and audited new seed 2440000.'},
        'regression_tests':int(count[1]),'write_publication_accepted':True,'learned_read_default_accepted':routing['enable_by_default'],
        'Chinese_training_started':False,'full_sleep_imagination_started':False}
    (ROOT/'reports/P12-P13-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rows=[]
    for row in memory['classifiers']:
        a=row['audit']['guarded_accuracy'];rows.append('| '+str(row['seed'])+' | '+' | '.join(f'{100*a[k]:.2f}%' for k in ('new','correct','support','repeat','defer'))+' |')
    worst_unknown=min(x['wrong_key_unknown'] for x in memory['memory'])
    portable=next(v for k,v in candidate['files'].items() if k.endswith('.pt'))
    ratio_min=min(1/r['speedup'] for r in routing['cases']);ratio_max=max(1/r['speedup'] for r in routing['cases'])
    med=cost['median_ms']
    text=f'''# P12/P13：内部记忆控制首批交付

2026-11-05。仅实施内部机制，中文与完整睡眠想象继续暂停。**来源与写入控制已通过有限任务验收；有预算读取只通过数值保持检查，性能未通过，默认不启用。** 原有个体保留，所有实验均在副本中进行。

## 实现与实际范围

- 新增5,189参数的随机初始化判定网络，区分新增、纠正、加强依据、重复、待定。置信度不足暂缓；来源为推测/模拟时，由结构保护禁止写当前事实。所有输入仍保留历史，不因不相信或重复而删除“收到过它”的经历。
- 当前依据及历史来源保存为关联神经矩阵，使用已有地址及RLS更新。来源类型、质量和有效事件时间都是环境给定的课程信息，不是模型已具备独立核实真假的能力。新矩阵属于有监督编码的快速权重，编码布局和更新算法是明确先验。
- 质量采用8位离散编码、有效时间采用10位编码。质量精度1/255；不足以可靠解码时拒绝提供数值。旧72次观察的来源保持未知，不回填假设。
- 已接入按新增信息数量与写满段触发的有限整理，单次最多尝试一段，继续使用P10行为验收及事务发布。不是后台常驻服务，也没有世界模型生成回放。
- 读取方面新增rank12得分近似和残差误差保护；训练因子先提议候选，不能证明遗漏误差足够小时完整读取。它不是自然语言语义寻址，也不是对未知未来段通用的路由器。

## 新审计流上的判断成绩

每个模型在相同的65,536组新数值输入上评估，不把重复使用的评估集算成不同样本集。列出保护与置信度门槛生效后的各类准确率：

| 种子 | 新增 | 纠正 | 加强依据 | 重复 | 待定 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

所有类别均达到95%门槛，但并非100%可靠。教师规则仅存在于训练脚本；作为简单规则基线，其本来就能准确执行这个有限政策。本轮没有证明神经实现比规则更快或更准确，目的在于为后续可学习的写入策略建立受控实验路径。

训练首版对少数类学习不足，“加强依据”的保护后准确率约90%，未发布。改为从训练专用262,144样本池按类别平衡抽取、余弦学习率，每个种子6,000步、3,060,000次样本展示；这些展示含重复，不能称为同量独立经历。三个基础种子均重新从随机初始化训练；原有骨干未修改。

## 连续经历、恢复与来源

三种子×两个数值世界，每条256次观察，共六条测试链。世界的事件类型顺序相同、具体值不同；不代表开放场景多样性。新增、纠正、重复、低质转述、过时信息、推测和模拟均包含在内。

- 六条链的当前值、原始历史、来源类别、已保存决策与预期行为均为100%；错键拒答最低{100*worst_unknown:.2f}%。
- 最终编码的有效事件时间最大误差0，质量最大误差0.001961，处于既定量化范围。归零来源矩阵后无法恢复该来源，恢复原矩阵后可重新读出。
- 每条链包含一次真实的历史蒸馏及多次便携保存/恢复，后续继续学习仍通过。
- 七个事务位置强制终止进程，完整读取、恢复与下一次写入一致；这不是物理断电试验。
- 接入时旧参数、快速状态、原回忆输出及P11计时模块均保持一致。11份已发布文件哈希不变。最终交付副本保留72次观察，未把演示输入写进去。

一次额外审视发现：初版来源类型正确，但连续回归的有效时刻最大偏差15步，质量平均偏差约0.013，因此撤回其发布资格。改用离散数值位后，以新审计种子2440000重跑全部写入/来源流程和恢复测试通过。原始记录、不带v2的试验文件保留为中间产物；最终请使用带v2的候选。这项修正没有提高判定网络的课程分数，也没有隐藏原误差。

## 读取的失败结果与成本

三种子各生成一条1,024次实际观察的历史，分别在4、8、16个填满段测量，历史已观察值在这些有限条件下均读对。它与来源模块的256次测试范围不同，不能声称多来源1,024次已经通过。每个条件使用128个随机数值查询，比较exact、P8有界路径及新方法。

九个随机查询条件的新路径全部回退，输出与全读差值为0，但总耗时约为exact的{ratio_min:.2f}–{ratio_max:.2f}倍。人为分离得分的对照可以减少读取且保持误差范围，但不计为应用收益。默认保持exact；发布副本不预装额外得分因子。

rank12因子每个区域增加8,640字节，另有运行计划、残差与矩阵副本；训练、构建和运行成本分别记录在完整数值中。评测只在CPU进行，未测峰值内存、能耗或长期尾部延迟。后续必须让查询与内容出现有意义的选择性，再验证是否值得启用。

新增判定权重与来源状态合计增加{candidate['tensor_bytes_after']-candidate['tensor_bytes_before']:,}字节载荷；便携文件由{candidate['parent_file_bytes']:,}增至{portable['bytes']:,}字节，增加{100*(portable['bytes']/candidate['parent_file_bytes']-1):.2f}%。每新增一段还要支付来源矩阵成本，重复输入也继续占历史容量。

本机两线程CPU、预热后的21次交错测量中：旧的直接可信写入中位{med['old_trusted_write']:.2f}ms，新增来源判断写入{med['sourced_write']:.2f}ms，读取来源{med['source_read']:.2f}ms。增加了判断开销；耗时受桌面负载影响，不作硬实时保证。一份增量提交样本写入{cost['commit']['tensor_bytes_written']:,}字节张量、{cost['commit']['manifest_bytes']:,}字节清单，另有数据库日志和页面开销。

## 文件与下一阶段

最终副本：`checkpoints/individual-internal-v2-candidate.sqlite`和`.pt`。源码、配置、文档和数值报告提交研究分支；权重、数据库和运行日志保留本机。{int(count[1])}项完整回归检查通过；相关10项检查也通过复核。

内部路线随后是快慢层语义巩固、再到有行动反馈的技能接口；两项目前是计划，不是已训练能力。来源可信度自主估计、多个独立证据合成、开放实体理解和语义路由仍未完成。完整睡眠想象与中文继续留在待办，外部工具系统不在本轮范围。

参见[实施计划](../docs/internal-memory-plan.md)、[实现及命令](../docs/internal-memory-implementation.md)与[完整数值及失败记录](P12-P13-results.json)。
'''
    (ROOT/'reports/P12-P13-report.md').write_text(text,encoding='utf-8')
    print(json.dumps({'write_passed':True,'routing_default':routing['enable_by_default'],'tests':int(count[1])}))


if __name__=='__main__':main()
