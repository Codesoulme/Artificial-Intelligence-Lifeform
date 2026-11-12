"""Stress-test natural-language observation routing without writing memory."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch

from scripts import deepseek_spora_natural_observation_transaction_pilot as pilot
from scripts import deepseek_spora_runtime_chat as runtime


OUTPUT = ROOT / 'runs/deepseek-spora-natural-route-stress-audit-20261113.json'

CASES = [
    ('positive_held', '据我所知，小梅最喜欢的饮品是茶。', True, '小梅', '茶'),
    ('positive_held', '小梅偏爱的饮料是咖啡。', True, '小梅', '咖啡'),
    ('positive_held', '阿远平时最爱喝牛奶。', True, '阿远', '牛奶'),
    ('positive_reworded', '小梅日常更愿意选择喝果汁。', True, '小梅', '果汁'),
    ('negative', '小梅不喜欢喝茶。', False, '小梅', None),
    ('negative', '小梅从来不喝咖啡。', False, '小梅', None),
    ('hypothesis', '如果小梅喜欢喝牛奶就好了。', False, '小梅', None),
    ('question', '你知道小梅喜欢喝什么吗？', False, '小梅', None),
    ('third_party', '小梅说阿远喜欢喝茶。', False, '小梅', None),
    ('action', '小梅给阿远买了咖啡。', False, '小梅', None),
    ('unknown_entity', '小雪偏爱的饮料是茶。', False, '小雪', '茶'),
    ('unknown_entity', '阿松不喜欢喝咖啡。', False, '阿松', None),
    ('unknown_value', '小梅偏爱的饮料是热巧克力。', False, '小梅', None),
    ('unknown_value', '阿远最喜欢喝白开水。', False, '阿远', None),
    ('multi_entity', '小梅喜欢喝茶，阿远喜欢喝牛奶。', False, '小梅', None),
    ('multi_entity', '小梅和阿远都喜欢喝咖啡。', False, '小梅', None),
    ('cross_predicate', '小梅喜欢什么颜色？', False, '小梅', None),
    ('cross_predicate', '小梅住在哪里？', False, '小梅', None),
]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    runtime.CHECKPOINT = pilot.CHECKPOINT
    runtime.WORKING_CHECKPOINT = ROOT / 'runs/deepseek-spora-output-v2-checkpoint-20261111/unused-stress.pt'
    chat = runtime.load_chat()
    components = pilot.build_components(chat)
    rows = []
    for group, text, expected_positive, expected_name, expected_value in CASES:
        parsed = pilot.parse_candidate(chat, text, components)
        parsed.pop('candidate')
        parsed.pop('gate')
        authorized = bool(parsed['address_ok'] and parsed['polarity_ok'] and parsed['value_ok'])
        name_present = expected_name in text
        value_exact = expected_value is None or parsed['value'] == expected_value
        rows.append({**parsed, 'group': group, 'expected_positive': expected_positive,
                     'expected_name': expected_name, 'expected_value': expected_value,
                     'name_present': name_present, 'value_exact': value_exact,
                     'authorized': authorized})
    positives = [row for row in rows if row['expected_positive']]
    negatives = [row for row in rows if not row['expected_positive']]
    summary = {
        'cases': len(rows),
        'positive_total': len(positives),
        'positive_address_ok': sum(row['address_ok'] for row in positives),
        'positive_polarity_ok': sum(row['polarity_ok'] for row in positives),
        'positive_value_ok': sum(row['value_ok'] for row in positives),
        'positive_authorized': sum(row['authorized'] for row in positives),
        'negative_total': len(negatives),
        'negative_authorized': sum(row['authorized'] for row in negatives),
        'negative_address_ok': sum(row['address_ok'] for row in negatives),
        'negative_polarity_ok': sum(row['polarity_ok'] for row in negatives),
        'unknown_entity_authorized': sum(row['authorized'] for row in rows if row['group'] == 'unknown_entity'),
        'unknown_value_authorized': sum(row['authorized'] for row in rows if row['group'] == 'unknown_value'),
        'multi_entity_authorized': sum(row['authorized'] for row in rows if row['group'] == 'multi_entity'),
        'cross_predicate_authorized': sum(row['authorized'] for row in rows if row['group'] == 'cross_predicate'),
    }
    report = {'status': 'natural_route_stress_audit_no_writes', 'summary': summary,
              'rows': rows, 'seconds': time.perf_counter() - started,
              'limitations': [
                  'Small authored challenge set, not a population sample.',
                  'Existing finite beverage routing artifacts and fixed entity enrollment.',
                  'No memory writes, no generated answers, and no open-domain parser training.',
              ]}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
