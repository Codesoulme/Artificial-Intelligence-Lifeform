"""Compare intent gates on a newly authored wording set without retraining."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_spora_runtime_chat import load_chat
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_read_intent_gate import OUTPUT as V2
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced
import torch

OUTPUT = ROOT/'runs/deepseek-spora-unseen-wording-audit-20261111.json'
V4 = ROOT/'runs/deepseek-spora-read-intent-gate-v4-20261111'

CASES = (
    ('小林的饮料偏好，在当前记录中是哪一个？', 0, 6),
    ('小林眼下保存的饮品选择是什么？', 0, 6),
    ('就现有资料看，小林喝哪种饮料？', 0, 6),
    ('小林现在留下的饮料偏好，请直接告诉我。', 0, 6),
    ('回到小林偏好改变之前，他喝的是什么？', 1, 1),
    ('小林第一次被记录为喜欢哪种饮品？', 1, 1),
    ('在偏好发生变化前，小林通常选择什么饮料？', 1, 1),
    ('小林早期那条饮品记录写的是什么？', 1, 1),
    ('偏好发生变化以后，小林改喝哪一种？', 2, 6),
    ('小林的饮料偏好更新后是什么结果？', 2, 6),
    ('在较新的记录里，小林选择了哪种饮品？', 2, 6),
    ('小林后来留下的饮料选择能查到吗？', 2, 6),
    ('小林的饮料资料来自谁？', None, None),
    ('请把小林的饮料偏好改写成牛奶。', None, None),
    ('小林和阿宁各自喝什么，请一起告诉我。', None, None),
    ('请为小林写一段生日祝福，不用查饮品。', None, None),
    ('小雪的旧饮料偏好是什么？', None, None),
    ('解释一下你是如何保存记忆的。', None, None),
)


def load_gate(path):
    report = json.loads((path/'result.json').read_text(encoding='utf-8'))
    checkpoint = path/'intent.pt'
    if __import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest() != report['intent_sha256']:
        raise ValueError('Intent gate digest mismatch')
    return torch.load(checkpoint, map_location='cpu', weights_only=True)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    chat = load_chat()
    v2_data = chat.intent_data
    v4_data = load_gate(V4)
    outcomes = {}
    for name, gate in (('v2', v2_data), ('v4', v4_data)):
        chat.intent_data = gate
        chat.state = fresh_controlled_state(chat.organism)
        for value in (1, 6):
            chat.state, decision = observe_sourced(chat.organism, chat.state, KEYS[0], value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled write rejected')
        rows = []
        for index, (question, expected_intent, expected_value) in enumerate(CASES):
            row = chat.answer(question)
            row.update({'index': index, 'expected_intent': expected_intent,
                        'expected_value': expected_value})
            row['route_correct'] = (expected_intent is None and not row['accepted']) or (
                expected_intent is not None and row['accepted']
                and row.get('temporal_intent') == expected_intent and row.get('key') == KEYS[0])
            row['answer_contains_expected'] = expected_value is not None and WORDS[expected_value] in (row['answer'] or '')
            rows.append(row)
        positives = [r for r in rows if r['expected_intent'] is not None]
        negatives = [r for r in rows if r['expected_intent'] is None]
        outcomes[name] = {'positive_total': len(positives),
                          'positive_routes_correct': sum(r['route_correct'] for r in positives),
                          'positive_answers_contains_expected': sum(r['answer_contains_expected'] for r in positives),
                          'negative_total': len(negatives),
                          'negative_rejected': sum(not r['accepted'] for r in negatives),
                          'rows': rows}
    report = {'status': 'unseen_authored_wording_gate_comparison', 'cases': len(CASES),
              'outcomes': outcomes,
              'limitations': ['Questions were authored for this audit and share the beverage domain.',
                              'One controlled entity and two finite facts; not an independent corpus.',
                              'The v4 candidate was designed after earlier development failures.',
                              'No open-domain Chinese conversation or autonomous observation is tested.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: {k: v for k, v in value.items() if k != 'rows'}
                      for name, value in outcomes.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
