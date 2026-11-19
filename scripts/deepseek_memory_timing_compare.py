"""Compare live memory handoff thresholds on regression and alternate phrasings."""

import argparse
import re
import time
from pathlib import Path

from deepseek_aligned_memory_dialogue import EARLY_TIMING_THRESHOLD, THRESHOLD
from deepseek_continuous_memory_chat import load_chat
from deepseek_memory_field_transfer import FACTS, QUESTIONS
from deepseek_output_aligned_memory import ROOT, fingerprint, write

OUT = ROOT / 'runs/deepseek-memory-timing-compare'
ALTERNATES = (
    (26, 'drink', '这里记了齐宁最中意哪种喝的。', '可以把这项念一遍吗？'),
    (47, 'drink', '刚才谈的是何程走的路线。', '我改问何程最喜欢哪种饮料。'),
    (26, 'work', '我在看齐宁的出行笔记。', '我想查齐宁具体干的是哪一行。'),
    (47, 'work', '这份材料提到了何程当时的职业。', '方便告诉我是什么吗？'),
    (26, 'place', '材料里写了齐宁那回到了什么地方。', '这个细节请帮我读一下。'),
    (47, 'place', '饮品偏好已经查完了。', '何程在那次出访里到的地点叫啥？'),
    (26, 'time', '我想知道齐宁到现场是几点。', '有写明具体答案吗？'),
    (47, 'ordinary', '何程那次去了哪儿刚说完。', '请用一句话招呼来参加聚会的朋友。'),
)
PREDICATES = {'drink': ('喝', '饮料是', '饮品是'), 'work': ('职业是', '工作是', '是一名'),
              'place': ('去了', '到访了', '地点是')}


def grade(result, field, expected, known_values):
    response = result['response']
    exact = (expected is not None and result['action'] == 'read' and result['field'] == field
             and response.rstrip('。.!！ ').endswith(expected))
    opening = response[:response.rfind(expected)].strip() if exact else ''
    repeat = bool(re.search(r'([\u3400-\u9fff]{1,4})\1$', opening))
    reasks = any(mark in opening for mark in ('？', '?'))
    declarative = exact and not repeat and not reasks and opening.endswith(PREDICATES[field])
    wrong_fact = any(value in response for value in known_values if value != expected)
    return dict(exact=exact, declarative=declarative, repeated_ending=repeat, reasks=reasks, wrong_fact=wrong_fact)


def main(out):
    out.mkdir()
    started = time.perf_counter()
    working = out / 'unused.pt'
    chat = load_chat(working, use_inherited_router=True, use_retrained_gate=True)
    sources = fingerprint((Path(__file__), ROOT / 'scripts/deepseek_continuous_memory_chat.py',
                           ROOT / 'scripts/deepseek_aligned_memory_dialogue.py'))
    known_values = {value for group in FACTS.values() for pair in group.values() for value in pair}
    suites = {'regression': [(entity, field, history, question) for entity, field, _, history, question in QUESTIONS],
              'alternate': ALTERNATES}
    results = {}
    for suite, cases in suites.items():
        rows = []
        for index, (entity, field, history, question) in enumerate(cases):
            versions = (0, 1) if field in FACTS[entity] else (1,)
            for version in versions:
                expected = FACTS[entity][field][version] if len(versions) == 2 else None
                for threshold in (THRESHOLD, EARLY_TIMING_THRESHOLD):
                    result = chat.answer(question, entity, version, history, timing_threshold=threshold)
                    rows.append(dict(case=index, expected=expected, threshold=threshold, requested_field=field,
                                     **result, **grade(result, field, expected, known_values)))
        results[suite] = rows
        print(f'Completed {suite}: {len(rows)} rows', flush=True)
    if fingerprint(tuple(Path(path) for path in sources)) != sources or working.exists():
        raise ValueError('Experiment inputs or memory changed during evaluation')
    write(out / 'result.json', dict(suites=suites, rows=results, sources=sources,
                                   elapsed_seconds=time.perf_counter() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
