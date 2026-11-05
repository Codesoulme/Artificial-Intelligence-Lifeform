"""Build a factual P1 report from completed local artifacts."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
seeds=[7,11,23]
audits=[json.loads((ROOT/f'runs/audit-seed{s}/audit.json').read_text(encoding='utf-8')) for s in seeds]
training=[json.loads((ROOT/f'runs/p1-seed{s}/summary.json').read_text(encoding='utf-8')) for s in seeds]
stress=json.loads((ROOT/'runs/stress-seed7.json').read_text(encoding='utf-8'))
report={
    'date':'2026-11-05','stage':'P1 exploratory prototype','training_seeds':seeds,
    'foundation_parameters':969576,'individual_fast_parameters':16640,
    'metrics':[{
        'seed':s,'steps':t['completed_steps'],'training_seconds':t['elapsed_seconds'],
        'peak_cuda_allocated_bytes':t['peak_cuda_allocated_bytes'],
        'scores':a['scores'],'paired_memory_gain':a['paired_memory_gain'],
        'paired_old_task_loss':a['paired_old_task_loss'],
        'restart_match':a['restart']['same_predictions'],'p1_gain_gate':a['p1_gain_gate'],
        'foundation_sha256':hashlib.sha256((ROOT/f'checkpoints/p1-seed{s}.pt').read_bytes()).hexdigest(),
    } for s,a,t in zip(seeds,audits,training)],
    'stress_seed7':stress,
    'limitations':['Simple symbolic associations, not Chinese conversation',
                  'No slow consolidation into P tested',
                  'No matched-compute learned alternative baseline yet',
                  'One-shot correction and finite-capacity retention fail',
                  'No evidence of artificial life or consciousness'],
}
out=ROOT/'reports';out.mkdir(exist_ok=True)
(out/'p1-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# SPORA P1 实践报告','',
       '2026-11-05。已完成本地最小神经记忆模型、有限训练和独立审计。', '',
       '## 实现与环境','',
       '基础网络 969,576 个参数；单个体另持有 16,640 个快速神经参数及对应动量。四层因果骨干、两个记忆接入点、随机初始化，无任何预训练权重。', '',
       'Python 3.11.9，PyTorch 2.5.1+cu121，GTX 1060 Max-Q 6GB。当前任务每次输入三个符号，输出 16 类之一；不把该耗时外推到语言训练。', '',
       '## 预设主实验','',
       '3 个训练起点各 1,000 步，每步 8 个世界、每世界 8 个事实，即各 64,000 个经历事件。审计在未参与训练或选型的 128 个新世界进行，每个 8 个初始事实，然后追加 8 个不同键的新事实。键与值的字母表已见，新组合未见。', '',
       '| 训练种子 | 写入后 | 无写入 | 追加后旧事实 | 追加后新事实 | 重启一致 |',
       '|---|---:|---:|---:|---:|---|']
for seed,a in zip(seeds,audits):
    score=a['scores']
    lines.append('| '+str(seed)+' | '+' | '.join(f'{score[k]["mean"]*100:.2f}%' for k in ['written','no_write','after_new_old','after_new_new'])+' | '+('是' if a['restart']['same_predictions'] else '否')+' |')
lines+=['','随机猜测期望为 6.25%。三个起点分别报告，不能将全部查询当作独立训练重复。世界层配对区间只是探索性估计，完整数值保存在同目录 JSON。','','## 参数来源验证','',
        '查询只传当前键，不拼接过去经历。将互斥随机事实写入不同快参数副本，结果跟随其经历；清空快速权重后增益消失。新进程载入个体文件后得到同样答案。', '',
        '重启程序在空工作目录运行，仅接收检查点路径和当前查询；答案由父评估进程保管。模型加载后安装文件与网络访问审计钩子，禁止查询期间外读。它属于实验隔离，不是操作系统安全沙箱。', '',
        '9 项测试通过，覆盖梯度、读写边界、独立世界、持久化、重启、恢复后继续学习、拒绝非有限权重和无效写入。另实测外循环断点继续；无效输入未破坏默认个体。', '',
        '## 主动寻找的失败','',
        '| 累积经历数 | 全部事实 | 最早 8 条 | 最近 8 条 |','|---|---:|---:|---:|']
for size,scores in stress['capacity'].items():
    lines.append('| '+size+' | '+' | '.join(f'{scores[k]*100:.2f}%' for k in ['all','oldest8','latest8'])+' |')
lines+=['',f'在 seed 7 的 32 个独立压力世界中，一次可信纠正成功率只有 {stress["single_write_correction"]["updated"]*100:.2f}%，其余旧事实正确率为 {stress["single_write_correction"]["unaffected"]*100:.2f}%。这些发现不用于此次重新调参。','',
        '解释：已经证明有限符号关联能通过神经快参数保存及重启读取；尚未解决事实修订、长序列干扰或慢速巩固。不能把 P1 通过改称完整持续学习成功。', '',
        '## 资源与本地文件','']
for seed,t in zip(seeds,training):
    lines.append(f'- seed {seed}：{t["elapsed_seconds"]:.1f} 秒；PyTorch 峰值活跃张量分配约 {t["peak_cuda_allocated_bytes"]/1024**2:.1f} MiB。')
lines+=['','以上显存不含驱动、CUDA 上下文、分配器保留与桌面占用；不是进程或整机总显存。','',
        '- `checkpoints/p1-seed7.pt`、`p1-seed11.pt`、`p1-seed23.pt`：基础学习结果及继续训练状态。',
        '- `checkpoints/individual.pt`：默认持续个体，以预先选定的 seed 7 为起点，已有演示经历 `12 → 5`。',
        '- `runs/audit-seed*/`：完整审计结果、世界得分和审计个体。',
        '- `runs/p1-seed*/`：配置、训练曲线与资源概要。','',
        '默认个体没有按审计结果挑选。其记忆体参与直接前向计算，没有携带事件查找表或身份文档。','',
        '## 下一步','',
        '下一轮优先研究可信纠正课程、顺序干扰和慢速巩固；保留独立新审计集及同计算预算基线。中文和结构增长应在这些机制获得证据后再推进。','']
(out/'P1-report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'report':str(out/'P1-report.md'),'all_gain_gates':all(a['p1_gain_gate'] for a in audits),
                  'all_restarts':all(a['restart']['same_predictions'] for a in audits)},ensure_ascii=False))
