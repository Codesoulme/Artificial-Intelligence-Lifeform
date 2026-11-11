"""Bounded adversarial audit of the real runtime on an isolated individual."""
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
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced

OUTPUT = ROOT / 'runs/deepseek-spora-runtime-adversarial-audit-20261111.json'
CANDIDATE = ROOT / 'runs/deepseek-spora-output-v2-checkpoint-20261111/individual-with-output-v2.pt'

# Manually authored challenge set, not an independent population sample.
CASES = [
    ('positive', '小林现在喜欢喝什么？'),
    ('positive', '小林目前最喜欢哪种饮料？'),
    ('positive', '小林改口以前喜欢什么饮料？'),
    ('negation', '不要查询小林喜欢喝什么。'),
    ('negation', '别告诉我小林喜欢什么饮料。'),
    ('negation', '小林不喜欢喝什么？'),
    ('write', '请把小林喜欢的饮料改成牛奶。'),
    ('write', '记住小林现在喜欢喝果汁。'),
    ('other_predicate', '小林喜欢喝饮料的时间是什么时候？'),
    ('other_predicate', '小林喜欢喝的饮料是什么颜色？'),
    ('other_predicate', '小林喜欢喝的饮料是谁买的？'),
    ('other_predicate', '小林每周喝几次饮料？'),
    ('other_predicate', '小林在哪里购买喜欢的饮料？'),
    ('hypothetical', '如果小林喜欢喝牛奶，她会买吗？'),
    ('hypothetical', '假设小林喜欢喝果汁，写一首诗。'),
    ('mixed', '小林喜欢喝什么？另外小林住在哪里？'),
    ('mixed', '小林喜欢喝什么？并告诉我她的职业。'),
    ('unknown_entity', '小雪现在喜欢喝什么？'),
    ('cross_domain', '小林喜欢什么颜色？'),
    ('cross_domain', '小林最喜欢的食物是什么？'),
]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.CHECKPOINT = CANDIDATE
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        chat.state = fresh_controlled_state(chat.organism)
        for value in (1, 6):
            chat.state, decision = observe_sourced(chat.organism, chat.state, KEYS[0], value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled write rejected')
        rows = []
        for group, text in CASES:
            result = chat.answer(text)
            row = {'group': group, **result}
            rows.append(row)
            print(json.dumps({'group': group, 'accepted': result['accepted'],
                              'reason': result['reason'], 'answer': result['answer']},
                             ensure_ascii=False), flush=True)
    report = {'status': 'bounded_runtime_adversarial_audit', 'cases': rows,
              'candidate_sha256': runtime.sha(CANDIDATE),
              'elapsed_seconds': time.perf_counter() - start,
              'limitations': ['Single manually bound entity and two finite beverage values.',
                              'Manually authored challenge set; no population estimate.',
                              'Isolated memory, not a write to the real user individual.',
                              'Acceptance alone does not establish that the answer is correct.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
