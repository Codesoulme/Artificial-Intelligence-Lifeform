"""Diversified suffix and multi-person prompts; development experiment only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts.deepseek_spora_boundary_train_pilot import digest, extract_names
from scripts.deepseek_spora_joint_span_crossed_pilot import build_records as crossed_records, TRAIN_TEMPLATES
from scripts.deepseek_spora_joint_span_pilot import main

OUTPUT = ROOT / 'runs/deepseek-spora-joint-span-diverse-pilot-20261113'
ADDED_SUFFIXES = ('娜', '明', '华')
UNSEEN_SUFFIXES = ('月', '兰')


def build_records(training, validation):
    train, evaluation, count = crossed_records(training, validation)
    names = sorted(extract_names(training) - extract_names(validation), key=digest)[:count]
    for name in names:
        for suffix in ADDED_SUFFIXES:
            for template in TRAIN_TEMPLATES[:2]:
                subject = name + suffix
                train.append({'question': template.format(name=subject), 'subject': subject, 'group': 'single'})
    for i in range(0, len(names), 2):
        a, b = names[i:i + 2]
        for question in (f'你记得{a}以及{b}爱喝什么吗？',
                         f'{a}与{b}各自通常喝什么？', f'请说出{a}和{b}的饮料喜好。'):
            train.append({'question': question, 'subject': None, 'group': 'multiple'})
    # Preserve the previous development set for a matched comparison, and add
    # a new suffix/wording combination; names and validation split are reused.
    previous = [r for r in evaluation if r['group'] == 'known_unseen']
    for row in previous:
        name = row['reference']
        subject = name + '兰'
        evaluation.append({'question': f'请问{subject}以前喜欢喝什么？', 'subject': subject,
                           'reference': name, 'group': 'suffix_lan_new_wording'})
    for i in range(0, 16, 2):
        names_eval = [r['reference'] for r in previous]
        a, b = names_eval[i:i + 2]
        evaluation.append({'question': f'{a}以及{b}的饮品分别是什么？',
                           'subject': None, 'group': 'multiple_new_wording'})
    return train, evaluation, count


if __name__ == '__main__':
    main(output=OUTPUT, build_records=build_records, steps=3200)
