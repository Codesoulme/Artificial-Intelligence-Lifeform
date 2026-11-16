"""Re-score a multi-relation bridge run without modifying its results."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from spora.research_relation_scoring import score_relation_answer

VALUE_RELATIONS = {**{v: '饮品' for v in ('茶', '咖啡', '牛奶', '豆浆', '果汁', '可乐')},
                   **{v: '颜色' for v in ('红色', '蓝色', '绿色', '黄色', '白色', '黑色')},
                   **{v: '活动' for v in ('读书', '跑步', '画画', '唱歌', '写字', '跳舞')}}
DEFAULT_RESULT = ROOT / 'runs/deepseek-spora-multirelation-hidden-bridge-64-20261117/result.json'


def score_phase(rows, value_relations=None):
    value_relations = VALUE_RELATIONS if value_relations is None else value_relations
    groups, pairs = defaultdict(lambda: {'correct': 0, 'total': 0}), {}
    for row in rows:
        condition = row['condition']
        if condition not in ('memory', 'swapped'):
            continue
        value = row['stored_expected']
        if value not in value_relations:
            raise ValueError('Unknown evaluated value')
        domain = value_relations[value]
        ok = score_relation_answer(row['answer'], row['name'], domain, value)
        key = (row['group'], domain, condition)
        groups[key]['total'] += 1
        groups[key]['correct'] += int(ok)
        pair_key = (row['group'], row['name'], row['question'], row['expected'])
        if condition in pairs.setdefault(pair_key, {}):
            raise ValueError('Duplicate counterfactual condition')
        pairs[pair_key][condition] = ok
    paired = [p for p in pairs.values() if set(p) == {'memory', 'swapped'}]
    return {'groups': [{'group': k[0], 'relation': k[1], 'condition': k[2], **v}
                       for k, v in sorted(groups.items())],
            'pair_total': len(paired),
            'pair_both_strict': sum(p['memory'] and p['swapped'] for p in paired)}


def main():
    if len(sys.argv) > 2:
        raise ValueError('Usage: script [result.json]')
    source = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_RESULT
    result = json.loads(source.read_text(encoding='utf-8'))
    print(json.dumps({'before': score_phase(result['before']), 'after': score_phase(result['after'])},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
