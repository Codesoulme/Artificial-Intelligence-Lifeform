"""Audit current runtime routing on fresh Chinese temporal formulations."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced


OUTPUT = ROOT / 'runs/deepseek-spora-runtime-temporal-generalization-audit-20261113.json'
CANDIDATE = ROOT / 'runs/deepseek-spora-output-v2-checkpoint-20261111/individual-with-output-v2.pt'

CASES = (
    ('positive', 0, 6, '小林眼下偏好的饮品是哪一种？'),
    ('positive', 0, 6, '现在小林通常喝什么？'),
    ('positive', 0, 6, '我想查一下小林目前的饮料偏好。'),
    ('positive', 0, 6, '小林此时最爱喝的东西是什么？'),
    ('positive', 0, 6, '请告诉我小林当下记录的饮品。'),
    ('positive', 0, 6, '小林最新留下的饮料是哪种？'),
    ('positive', 1, 1, '小林在改口之前偏爱什么饮料？'),
    ('positive', 1, 1, '较早的时候，小林爱喝哪一类饮品？'),
    ('positive', 1, 1, '小林最初记录的饮料偏好是什么？'),
    ('positive', 1, 1, '请翻出小林变更前的饮品信息。'),
    ('positive', 1, 1, '小林早先常喝的是哪种？'),
    ('positive', 1, 1, '在更新以前，小林喜欢什么饮料？'),
    ('positive', 2, 6, '小林改了偏好以后选择了什么饮料？'),
    ('positive', 2, 6, '后来的记录里小林爱喝什么？'),
    ('positive', 2, 6, '小林在那次更正之后偏爱哪种饮品？'),
    ('positive', 2, 6, '更新完成后，小林的饮料选择是什么？'),
    ('positive', 2, 6, '小林后来改成喜欢喝什么了？'),
    ('positive', 2, 6, '再往后的记录中，小林最爱哪种饮料？'),
    ('negative', None, None, '请给小林写一句鼓励的话。'),
    ('negative', None, None, '不要查询小林的饮料记录。'),
    ('negative', None, None, '把小林喜欢的饮料改成牛奶。'),
    ('negative', None, None, '小林和阿宁分别喜欢喝什么？'),
    ('negative', None, None, '小雪以前喜欢喝什么？'),
    ('negative', None, None, '你能解释一下社会主义核心价值观吗？'),
    ('negative', None, None, '小梅后来喝什么？'),
    ('negative', None, None, '小林的饮品记录是谁写下的？'),
    ('negative', None, None, '帮我设计一个饮料广告。'),
    ('negative', None, None, '小林现在喜欢什么以及阿宁过去喜欢什么？'),
    ('negative', None, None, '请把小林和小雨的偏好都查出来。'),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not CANDIDATE.exists():
        raise FileNotFoundError(CANDIDATE)
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.CHECKPOINT = CANDIDATE
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        chat.state = fresh_controlled_state(chat.organism)
        for value in (1, 6):
            chat.state, decision = observe_sourced(chat.organism, chat.state, KEYS[0], value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled memory write rejected')
        rows = []
        for group, expected_mode, expected_value, question in CASES:
            result = chat.answer(question)
            answer = result.get('answer') or ''
            answer_correct = (expected_value is not None and WORDS[expected_value] in answer)
            route_correct = result.get('accepted') is (group == 'positive')
            if group == 'positive':
                route_correct = route_correct and result.get('temporal_intent') == expected_mode
            rows.append({'group': group, 'question': question,
                         'expected_mode': expected_mode, 'expected_value': expected_value,
                         'answer_correct': answer_correct, 'route_correct': route_correct,
                         **result})
    report = {
        'status': 'current_runtime_temporal_generalization_audit',
        'case_count': len(CASES),
        'positive_total': sum(group == 'positive' for group, *_ in CASES),
        'positive_route_correct': sum(row['route_correct'] for row in rows if row['group'] == 'positive'),
        'positive_answer_correct': sum(row['answer_correct'] for row in rows if row['group'] == 'positive'),
        'negative_total': sum(group == 'negative' for group, *_ in CASES),
        'negative_rejected': sum(not row['accepted'] for row in rows if row['group'] == 'negative'),
        'rows': rows,
        'candidate_sha256': runtime.sha(CANDIDATE),
        'elapsed_seconds': time.perf_counter() - started,
        'limitations': [
            'Manually authored formulations and one bound entity.',
            'Finite beverage vocabulary and controlled two-event history.',
            'Does not establish open-world Chinese chat or autonomous observation.',
        ],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in (
        'positive_total', 'positive_route_correct', 'positive_answer_correct',
        'negative_total', 'negative_rejected')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
