"""Verify the Windows launcher with real writes and a separate-process restart."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess
from time import perf_counter

from scripts.train_spora_recurrent_counterfactual import store_json


ROOT = Path(__file__).resolve().parents[1]
VALUES = {1: '果茶', 2: '枸杞茶', 3: '绿豆汤', 4: '银杏亭', 5: '茶', 6: '山海志'}
WRITE_AND_CHAT = (
    ('help', '/帮助', ()),
    ('saved', '/记 沈予安 | 饮品 | 果茶', (1,)),
    ('saved', '/记 沈予安 | 饮品 | 枸杞茶', (2,)),
    ('saved', '/记 沈予安 | 饮品 | 绿豆汤', (3,)),
    ('saved', '/记 沈予安 | 见面地点 | 银杏亭', (4,)),
    ('saved', '/记 沈予岚 | 饮品 | 茶', (5,)),
    ('saved', '/记 孟景纾 | 书目 | 山海志', (6,)),
    ('error', '/记 沈予安 | 饮品 | 这是一段故意写得很长并且超过限制的值', ()),
    ('read', '请查沈予安的饮品当前版。', (3,)),
    ('read', '那上一版呢？', (2,)),
    ('read', '那首版呢？', (1,)),
    ('clarify', '请查沈予安的饮品首版还是现行版？', ()),
    ('read', '查当前版。', (3,)),
    ('chat', '换个话题，怎样把书桌收拾整齐？', ()),
    ('chat', '只说一个立刻能做的办法。', ()),
    ('clarify', '那上一版呢？', ()),
    ('read', '请查沈予安的饮品当前版；查沈予安的见面地点当前版。', (3, 4)),
    ('clarify', '两项都查上一版。', ()),
    ('read', '请查沈予安的见面地点当前版。', (4,)),
    ('chat', '把「那上一版呢」翻译成英语。', ()),
    ('clarify', '那首版呢？', ()),
    ('read', '请查沈予岚的饮品当前版。', (5,)),
    ('clarify', '那沈予舟的饮品上一版呢？', ()),
    ('clarify', '那首版呢？', ()),
    ('read', '查孟景纾的书目当前版。', (6,)),
    ('reset', '/清空对话', ()),
    ('clarify', '那首版呢？', ()),
)
AFTER_RESTART = (
    ('clarify', '那上一版呢？', ()),
    ('read', '请查沈予安的饮品当前版。', (3,)),
    ('read', '那上一版呢？', (2,)),
    ('read', '那首版呢？', (1,)),
    ('read', '请查沈予岚的饮品当前版；查沈予安的见面地点当前版。', (5, 4)),
    ('defer', '先别查，沈予安的饮品上一版我过会儿再问。', ()),
    ('clarify', '那首版呢？', ()),
    ('read', '查孟景纾的书目首版。', (6,)),
)


def replay(profile, cases):
    started = perf_counter()
    process = subprocess.run(['cmd.exe', '/d', '/c', 'spora_version_chat.cmd', '--profile', str(profile), '--json'],
                             cwd=ROOT, input='\n'.join([*(question for _, question, _ in cases), '/退出', '']),
                             encoding='utf-8', capture_output=True, timeout=270)
    if process.returncode:
        raise RuntimeError(process.stderr)
    rows = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    checks = []
    for (kind, question, selected), result in zip(cases, rows, strict=True):
        actual = (result['record'],) if result['kind'] == 'saved' else tuple(result.get('selected', ()))
        expected_values = [VALUES[number] for number in selected] if kind == 'read' else []
        answer = result['answer']
        positions = [answer.find(value) for value in expected_values]
        foils = [value for number, value in VALUES.items() if number not in selected
                 and not any(value in expected for expected in expected_values)]
        reply_exact = (not expected_values or all(position >= 0 for position in positions)
                       and positions == sorted(positions) and answer[-1:] in '。！？.!?'
                       and not any(foil in answer for foil in foils))
        passed = result['kind'] == kind and actual == selected and reply_exact
        checks.append({'question': question, 'expected_kind': kind, 'expected_records': list(selected),
                       'actual': result, 'passed': passed})
        if not passed:
            print('failed', checks[-1], flush=True)
    return {'checks': checks, 'seconds': perf_counter() - started, 'passed': sum(row['passed'] for row in checks)}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    profile = (args.output / 'profile').resolve()
    first = replay(profile, WRITE_AND_CHAT)
    database = profile / 'versions.sqlite3'
    before = sha256(database.read_bytes()).hexdigest()
    after = replay(profile, AFTER_RESTART)
    if sha256(database.read_bytes()).hexdigest() != before:
        raise ValueError('The restart modified saved memory during read-only turns')
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        rows = connection.execute('SELECT number,name,field,effective,observed FROM versions ORDER BY number').fetchall()
    if len(rows) != len(VALUES) or len({row[3] for row in rows[:3]}) != 1:
        raise ValueError('Expected three distinct versions recorded for one effective day')
    result = {'first_process': first, 'restart': after, 'database_sha256': before, 'record_count': len(rows),
              'database_bytes': database.stat().st_size}
    store_json(args.output / 'result.json', result)
    print(f"first {first['passed']}/{len(WRITE_AND_CHAT)}, restart {after['passed']}/{len(AFTER_RESTART)}", flush=True)
    if first['passed'] != len(WRITE_AND_CHAT) or after['passed'] != len(AFTER_RESTART):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args())
