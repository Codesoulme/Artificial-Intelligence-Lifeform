"""Probe temporal heads on new names and several non-beverage predicates."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_temporal_gate_v5_contrast as candidate
from scripts import deepseek_spora_temporal_v2_holdout as audit

audit.trainer.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-gate-v5-contrast-20261112'
audit.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-cross-domain-audit-20261112.json'
# Keys are identifiers for this head-only probe, not claims of name binding.
audit.ENROLLMENTS = ((6, '小林', 0), (1001, '岑砚之', 0), (1002, '黎照宁', 0))
audit.TEMPLATES = {
    0: ('{name}目前住在哪座城市？', '过去的工作先不谈，{name}现在的职业是什么？',
        '眼下{name}正在读哪本书？', '现在{name}习惯哪种运动？', '请问{name}当前喜欢什么颜色？'),
    1: ('搬家之前{name}住在哪座城市？', '不问现在，请说{name}原先做什么工作。',
        '换书之前{name}在读哪一本？', '后来发生的改变不用说，{name}早先练什么运动？',
        '更换偏好前{name}喜欢哪种颜色？'),
    2: ('搬家之后{name}住在哪座城市？', '不问过去，请说{name}后来做什么工作。',
        '换书之后{name}在读哪一本？', '以前的情况不用说，{name}后来练什么运动？',
        '更换偏好后{name}喜欢哪种颜色？'),
    3: ('{name}搬家前后分别住在哪里？', '不要告诉我{name}目前的职业。',
        '如果{name}换了书，请写个故事。', '修改{name}过去的运动记录。',
        '{name}是哪天换了喜欢的颜色？'),
}


def main():
    audit.main()
    report = json.loads(audit.OUTPUT.read_text(encoding='utf-8'))
    report['status'] = 'cross_domain_temporal_head_development'
    report['limitations'] = [
        'One known name and two names not found by local literal search; no identity binding tested.',
        'Five domains and twenty authored templates; names repeat the same templates.',
        'Exact template overlap checked against current candidate additions only, not all training or history.',
        'Head-only test: non-beverage predicates are not supported by the default runtime memory path.',
        'Multi-time, negated, hypothetical, edit and date requests labeled abstain for this finite task.',
        'Not a blind independent benchmark or proof of real memory generalization.',
    ]
    groups = {}
    for name in ('小林', '岑砚之', '黎照宁'):
        subset = [r for r in report['rows'] if name in r['question']]
        groups[name] = {'cases': len(subset),
                        'baseline_correct': sum(r['baseline'] == r['label'] for r in subset),
                        'candidate_correct': sum(r['candidate'] == r['label'] for r in subset),
                        'baseline_wrong_read': sum(r['baseline'] != 3 and r['baseline'] != r['label'] for r in subset),
                        'candidate_wrong_read': sum(r['candidate'] != 3 and r['candidate'] != r['label'] for r in subset)}
    report['name_groups'] = groups
    report['common_wrong_reads'] = [r for r in report['rows'] if r['baseline'] == r['candidate'] != 3
                                    and r['baseline'] != r['label']]
    audit.OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'names': groups, 'common_wrong_reads': len(report['common_wrong_reads'])}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
