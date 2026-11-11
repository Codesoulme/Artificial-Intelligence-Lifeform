"""Broader authored contrast set for temporal heads and agreement heuristic."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_temporal_v2_holdout as audit

audit.trainer.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-gate-v3-20261112'
audit.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-contrast-audit-20261112.json'
audit.TEMPLATES = {
    0: ('只查当前{name}喜欢的饮料，不要以前的。', '今天{name}喜欢喝哪一种饮料？',
        '现在{name}的饮料偏好是什么？', '请说{name}目前钟爱的饮料。',
        '历史先不管，{name}现在偏好哪种饮品？', '眼下{name}选的饮料是？'),
    1: ('只查{name}改动前喜欢的饮料，不要现在的。', '更新前{name}喝的饮品是什么？',
        '后来的变化先不管，{name}先前喜欢哪种饮料？', '回顾更改之前{name}钟爱的饮品。',
        '{name}过去的饮料选择是什么，不是现在那项？', '请说{name}改过饮料偏好之前喜欢什么。'),
    2: ('只查{name}改动后喜欢的饮料，不要以前的。', '更新后{name}喝的饮品是什么？',
        '以前的选择先不管，{name}后来喜欢哪种饮料？', '回顾更改之后{name}钟爱的饮品。',
        '{name}后来的饮料选择是什么，不是先前那项？', '请说{name}改过饮料偏好之后喜欢什么。'),
    3: ('{name}以前和现在分别喜欢什么饮料？', '别告诉我{name}过去喜欢的饮料。',
        '如果{name}改喝茶，写一个小故事。', '更新{name}过去饮料的那条记录。',
        '{name}的饮料偏好是哪一天改变的？', '说说{name}的饮料偏好，不用区分时间。'),
}


def main():
    audit.main()
    report = json.loads(audit.OUTPUT.read_text(encoding='utf-8'))
    report['status'] = 'authored_temporal_contrast_development'
    report['limitations'].extend([
        'Previously used models and design informed the new templates; not an independent unbiased benchmark.',
        'Ambiguous or multi-time requests were assigned abstention for this single-time experiment.',
        'Agreement uses observed model decisions and is not a calibrated error guarantee.',
    ])
    aggregate = {}
    for mode in ('baseline', 'candidate', 'agreement'):
        rows = report['rows']
        for row in rows:
            row['agreement'] = row['baseline'] if row['baseline'] == row['candidate'] else 3
        aggregate[mode] = {'correct': sum(r[mode] == r['label'] for r in rows),
                           'wrong_read': sum(r[mode] != 3 and r[mode] != r['label'] for r in rows),
                           'false_abstain': sum(r[mode] == 3 and r['label'] != 3 for r in rows),
                           'read': sum(r[mode] != 3 for r in rows), 'cases': len(rows)}
    report['agreement_comparison'] = aggregate
    report['common_wrong_reads'] = [r for r in report['rows']
                                    if r['baseline'] == r['candidate'] != 3
                                    and r['baseline'] != r['label']]
    audit.OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'comparison': aggregate, 'common_wrong_reads': len(report['common_wrong_reads'])}), flush=True)


if __name__ == '__main__':
    main()
