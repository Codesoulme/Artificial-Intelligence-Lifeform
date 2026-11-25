"""Replay typed memory references and time clarification through the Windows entry."""

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

from scripts.audit_spora_local_composition import RECORDS, ROOT, WRITES
from scripts.audit_spora_fresh_dialogue import digest
from scripts.train_spora_recurrent_counterfactual import store_json


PAIR = '请查许林秋的见面地点当前版；查许林秋的用品上一版。'
CURRENT = PAIR.replace('用品上一版', '用品当前版')
REQUEST = '按这两项写一句消息，说明在哪儿碰面，还要提醒对方带上要带的东西。'
INLINE = REQUEST.removesuffix('。') + '。时间定在周五傍晚。'
AMBIGUOUS = REQUEST + '时间定在周五傍晚或者星期三中午。'
BOOK_PAIR = '请查姚清婉的见面地点当前版；查姚清婉的书目首版。'
BOOK_REQUEST = '按这两项写一句消息，约见面，提醒对方带上那本书。时间定在今晚七点。'
FIRST = (
    *WRITES[:8],
    (REQUEST, 'clarify', ()),
    (PAIR, 'read', (2, 3)),
    (REQUEST, 'clarify', ()),
    ('周五傍晚。', 'composed', (2, 3)),
    (CURRENT, 'read', (2, 4)),
    (INLINE, 'composed', (2, 4)),
    (BOOK_PAIR, 'read', (8, 5)),
    (BOOK_REQUEST, 'composed', (8, 5)),
    (PAIR, 'read', (2, 3)),
    (REQUEST, 'clarify', ()),
)
SECOND = (
    (REQUEST, 'clarify', ()),
    (PAIR, 'read', (2, 3)),
    (REQUEST, 'clarify', ()),
    ('周五傍晚。', 'composed', (2, 3)),
    (CURRENT, 'read', (2, 4)),
    (INLINE, 'composed', (2, 4)),
    (PAIR, 'read', (2, 3)),
    (AMBIGUOUS, 'clarify', ()),
    ('换个话题，用一句话说怎么擦洗竹篮。', 'chat', ()),
    (REQUEST, 'clarify', ()),
    (PAIR, 'read', (2, 3)),
    (REQUEST, 'clarify', ()),
    ('算了。', 'defer', ()),
    (REQUEST, 'clarify', ()),
    (PAIR, 'read', (2, 3)),
    (REQUEST, 'clarify', ()),
    ('/清空对话', 'reset', ()),
    (REQUEST, 'clarify', ()),
    ('请查许林秋的用品当前版；查姚清婉的书目首版。', 'read', (4, 5)),
    (REQUEST, 'clarify', ()),
)


def replay(profile, cases, composition):
    start = perf_counter()
    proc = subprocess.run(
        ['cmd.exe', '/d', '/c', 'spora_version_chat.cmd', '--profile', str(profile),
         '--composition', str(composition), '--json'],
        cwd=ROOT, input='\n'.join([*(line for line, _, _ in cases), '/退出', '']),
        encoding='utf-8', capture_output=True, timeout=270,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    output = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    observed = []
    for (question, kind, expected), result in zip(cases, output, strict=True):
        selected = (result['record'],) if result['kind'] == 'saved' else tuple(result.get('selected', ()))
        passed = kind == result['kind'] and expected == selected
        if kind == 'composed':
            wanted = {RECORDS[number - 1][2] for number in expected}
            shown = {value for _, _, value in RECORDS if value in result['answer']}
            when = '今晚七点' if question == BOOK_REQUEST else '周五傍晚'
            passed &= wanted == shown and result['answer'].count(when) == 1 and len(result.get('copied', ())) == 3
        row = {'question': question, 'expected': kind, 'expected_records': expected, 'output': result, 'passed': passed}
        observed.append(row)
        if not passed:
            print(row, flush=True)
    return {'checks': observed, 'seconds': perf_counter() - start, 'passed': sum(row['passed'] for row in observed)}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    profile = (args.output / 'profile').resolve()
    first = replay(profile, FIRST, args.composition)
    database = profile / 'versions.sqlite3'
    original = digest(database)
    second = replay(profile, SECOND, args.composition)
    initial = {tuple(row['expected_records']): row['output']
               for row in first['checks'] if row['expected'] == 'composed'}
    after = {tuple(row['expected_records']): row['output']
             for row in second['checks'] if row['expected'] == 'composed'}
    checks = {'archive_unchanged': original == digest(database),
              'same_composition_after_restart': bool(after) and all(initial[number] == value for number, value in after.items())}
    store_json(args.output / 'result.json', {'first': first, 'restart': second, 'controls': checks})
    print(f"first {first['passed']}/{len(FIRST)}, restart {second['passed']}/{len(SECOND)}", checks, flush=True)
    if first['passed'] != len(FIRST) or second['passed'] != len(SECOND) or not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--composition', type=Path, default=Path('runs/spora-typed-composition-v1'))
    main(parser.parse_args())
