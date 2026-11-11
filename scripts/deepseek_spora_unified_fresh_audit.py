"""Audit the unified persisted runtime on the fresh authored wording set."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_spora_independent_route_audit import CASES
from scripts.deepseek_spora_runtime_chat import load_chat
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced

OUTPUT = ROOT/'runs/deepseek-spora-unified-fresh-audit-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    chat = load_chat()
    chat.state = fresh_controlled_state(chat.organism)
    for value in (1, 6):
        chat.state, decision = observe_sourced(chat.organism, chat.state, KEYS[0], value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled audit write rejected')
    rows = []
    for index, case in enumerate(CASES):
        row = chat.answer(case['text'])
        row.update({'index': index, 'expected_intent': case['label'], 'expected_value': case.get('value')})
        expected = case.get('value')
        row['answer_contains_expected'] = expected is not None and WORDS[expected] in (row['answer'] or '')
        row['route_correct'] = (case['label'] is None and not row['accepted']) or (
            case['label'] is not None and row['accepted']
            and row.get('temporal_intent') == case['label'] and row.get('key') == KEYS[0])
        rows.append(row)
        print(json.dumps({'index': index, 'accepted': row['accepted'],
                          'route_correct': row['route_correct'],
                          'answer_contains_expected': row['answer_contains_expected']}), flush=True)
    positives = [r for r in rows if r['expected_intent'] is not None]
    negatives = [r for r in rows if r['expected_intent'] is None]
    report = {'status': 'unified_persisted_runtime_fresh_wording_audit',
              'positive_total': len(positives),
              'positive_routes_correct': sum(r['route_correct'] for r in positives),
              'positive_answers_contains_expected': sum(r['answer_contains_expected'] for r in positives),
              'negative_total': len(negatives),
              'negative_rejected': sum(not r['accepted'] for r in negatives),
              'rows': rows,
              'limitations': ['Fresh authored questions share the narrow beverage domain.',
                              'One controlled entity and two finite facts; writes are supplied by the audit.',
                              'Word containment is a coarse answer metric.',
                              'This does not establish open-domain Chinese conversation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('positive_total', 'positive_routes_correct',
                                             'positive_answers_contains_expected', 'negative_total',
                                             'negative_rejected')}, ensure_ascii=False), flush=True)


from scripts.deepseek_semantic_value_pilot import WORDS


if __name__ == '__main__':
    run()
