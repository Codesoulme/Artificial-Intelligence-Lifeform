"""Publish measured P11 evidence; weights and run artifacts remain local."""
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_time_runtime import protected_files


def main():
    protected_files()
    read = lambda name: json.loads((ROOT/name).read_text(encoding='utf-8'))
    neural = read('runs/p11-time-world-audit/audit.json')
    runtime = read('runs/p11-runtime-audit-v2/audit.json')
    first = read('runs/p11-runtime-audit/audit.json')
    candidate = read('runs/p11-candidate.json')
    tests = (ROOT/'runs/p11-final-regression.log').read_text(encoding='utf-8')
    count = re.search(r'Ran (\d+) tests', tests)
    if not count or not tests.rstrip().endswith('OK') or not all(d['passed'] for d in (neural, runtime, candidate)):
        raise RuntimeError('Incomplete or failed final checks')
    report = {'date': '2026-11-05', 'parent_commit': '327328299d9f290cf4a77d33a3efcde7458168de',
              'neural': neural, 'runtime': runtime, 'candidate': candidate,
              'initial_audit_failure': {'passed': first['passed'], 'real_run': first['real_run'],
                  'cause': 'Audit incorrectly applied the idle-only <=12 byte threshold to initial observations and incoming pulses.',
                  'fix': 'Classify commits by newly observed input counts; retain <=12 bytes for idle, <=48 bytes for new input. Original neural gates unchanged.',
                  'clock_refinement': 'Use monotonic high-resolution perf_counter_ns rather than lower-resolution Windows monotonic_ns.'},
              'regression_tests_passed': int(count[1]), 'Chinese_training_started': False,
              'passed': True}
    (ROOT/'reports/P11-results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = []
    for entry in neural['seeds']:
        a = entry['audit']
        rows.append(f"| {entry['seed']} | "+' / '.join(f'{v:.5f}' for v in a['rmse'])
                    +f" | {100*a['period_relative_mae']:.3f}% | {a['time_ablation_error_ratio']:.2f}倍 |")
    real = runtime['real_run']
    portable = next(v for k, v in candidate['files'].items() if k.endswith('.pt'))
    text = f'''# P11 交付：真实时长与有限环境预测

2026-11-05。阶段验收通过，中文训练仍为0。本轮交付的是可保存、可恢复的时间与简单动力学能力，不构成主观体验或人工生命的证据。

## 完成内容

从随机初始化训练三个小模块，分别使用种子7、11、23，每个2,000步、512,000个合成环境转移及同量周期样本。没有使用预训练参数。按审计前确定的seed7接入已存在的睡眠候选副本，新增4,932个参数，原有神经张量和72次符号观察保留。

在线时长由单调高精度时钟提供；循环次数、用户消息次数与秒数分开。小网络预测规则信号间隔，以及位置、速度、温差在给定动作下的变化。重启时保持已提交状态，空白期间只作时间估计，并要求新观察后才恢复环境预测。UTC倒退不减少累计在线值。

## 独立数据结果

三个种子使用同一份未参与训练的4,096组随机环境和4,096组周期样本；不是12,288组互不重复的环境。环境参数已知且直接输入，动作在每个预测窗内保持不变。误差单位是合成环境的数值单位。

| 种子 | 位置 / 速度 / 温差 RMSE | 周期相对平均绝对误差 | 固定时长消融相对误差 |
| --- | --- | --- | --- |
{chr(10).join(rows)}

预设门槛为各分量RMSE≤0.05、周期误差≤3%、固定时长消融误差至少翻倍，三组均通过。另按0.1、0.5、1、2秒分别检验，最差分量RMSE为0.03185以内。正反动作方向对照全部通过；将动作置零明显恶化结果。完整分量与对照见[P11-results.json](P11-results.json)。

**没有证明神经周期估计优于简单方法。** 同数据的“两次间隔取平均”误差为0.674%，比三个网络都低。此处证明模块能学习该任务，不能据此宣称结构创新或优越性。世界预测仅覆盖给定课程；超过2秒返回需要观察。

## 运行、恢复与兼容性

- 相同总时长拆为1次或100次推进，三组预测最大差值均为0；不同总时长会改变结果。
- 独立进程实际运行{real['wall_seconds']:.3f}秒，执行{real['ticks']}次更新。调度器发送三次合成信号，在约{real['events'][0]['elapsed_seconds']:.3f}秒产生一次预计到期提示。该提示没有被写为第四次观察；符号游标始终为72。
- 起点只接收一个合成世界状态；超过2秒后明确要求新观察。没有真实传感器接入，也没有将预测当成已发生事件。
- 同时改变时间权重、工作状态、时钟元数据及符号游标，在七个事务位置强制结束进程。并发读取、重启恢复、下一次写入均精确一致。此为进程终止测试，不是物理断电试验。
- 接入后、空闲运行后，所有64个键的当前输出，以及两段全部时间位置的历史输出，均与父个体逐值一致。便携和数据库恢复也逐值一致。
- 审计副本继续观察至128次，再对第二段做600步睡眠蒸馏，已知/未知分组行为一致率均100%；新参数和工作状态精确保留。该延续只发生在审计副本，不写入发布个体。
- {int(count[1])}项完整回归检查通过，原有九份受保护文件的校验值全部不变。

## 成本与一次验收修正

新增参数占19,728字节，固定传感工作张量占48字节，共增加19,776字节张量载荷。发布便携文件由{candidate['parent_portable_bytes']:,}增至{portable['bytes']:,}字节，约增加{100*(portable['bytes']/candidate['parent_portable_bytes']-1):.2f}%。时钟计数和清单另占元数据空间。空闲循环不增长历史段或参数数量。

本机CPU使用两个PyTorch线程，更新平均{real['tick_ms_mean']:.2f}毫秒、最大{real['tick_ms_max']:.2f}毫秒；本次有界运行消耗进程CPU时间{real['cpu_seconds']:.3f}秒。此统计不包括解释器启动和首次加载；只有一次短测，不能用作长期性能保证，未测峰值进程内存或能耗。预测从最近观察重新积分，在2秒窗口内计算量随时长增加，上限由积分步长和窗口限制。

首次观察提交48字节张量，后续脉冲与预测一起提交20字节；没有新观察的更新为12字节或0字节。每次仍需约15.7KB清单，并产生SQLite页面和日志开销，所以不等于仅写12字节磁盘。

第一版运行审计错误地把空闲更新的12字节上限套到了首次观察，导致总验收为false。修正为按新增输入计数区分保存类型，保留原日志和失败记录，再完整重跑通过；没有放宽神经误差、恢复或旧记忆门槛。同时换用Windows下更精细的单调性能计数器。失败记录保留在机器可读结果中。

## 交付与尚未完成

本地发布文件为 `checkpoints/individual-time-candidate.sqlite` 和 `.pt`，版本0，仍保留72次符号观察；工作状态初始未观察。真实等待和后续睡眠测试都在独立副本运行。源码、设计补充、配置、报告及测试推送研究分支，权重和数据库继续保留本机，不纳入Git。

新模块学习的是通用课程规律；最近数值观察明确保存在工作张量中，不伪称每个时间数值都写入长期神经记忆。没有完整的循环自我状态、未知参数推断、语言时间理解、主动感知策略或开放世界推演。预测模块尚未与睡眠想象回放联训，P3慢速专家也未合入。当前阶段可以继续研究，但不能宣布完整做梦系统或生命结构已完成。

实现、操作命令和研究对应关系见[设计补充](../docs/time-world-implementation.md)。本次运行已经退出，没有留下常驻训练或后台计时服务。下一阶段建议先做不规则事件概率与误差校准，再把有边界的短时想象接入睡眠核对；中文仍需保持暂停，直至另行决定。
'''
    (ROOT/'reports/P11-report.md').write_text(text, encoding='utf-8')
    print(json.dumps({'passed': True, 'tests': int(count[1]), 'portable_bytes': portable['bytes']}))


if __name__ == '__main__':
    main()
