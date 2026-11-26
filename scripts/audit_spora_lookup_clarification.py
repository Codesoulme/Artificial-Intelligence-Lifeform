"""Exercise partial lookup, correction, and cancellation using a real local archive."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess
from time import perf_counter

from scripts.audit_spora_version_local_chat import ROOT
from scripts.train_spora_recurrent_counterfactual import store_json


VALUES = {
    1: '果茶', 2: '枸杞茶', 3: '银杏亭', 4: '白鹭堂', 5: '青瓷杯', 6: '灰书签', 7: '茉莉茶',
}
WRITE = (
    ('/记 沈予安 | 饮品 | 果茶', 'saved', 1),
    ('/记 沈予安 | 饮品 | 枸杞茶', 'saved', 2),
    ('/记 沈予安 | 见面地点 | 银杏亭', 'saved', 3),
    ('/记 沈予安 | 见面地点 | 白鹭堂', 'saved', 4),
    ('/记 沈予安 | 用品 | 青瓷杯', 'saved', 5),
    ('/记 沈予岚 | 用品 | 灰书签', 'saved', 6),
    ('/记 沈予岚 | 饮品 | 茉莉茶', 'saved', 7),
)
FIRST = (
    *WRITE,
    ('请查谁的用品当前版。', 'clarify', '谁的用品'),
    ('沈予岚。', 'read', 6),
    ('查一下沈予安的上一版。', 'clarify', '哪项记录'),
    ('见面地点。', 'read', 3),
    ('查沈予安的记录。', 'clarify', '哪项记录'),
    ('不是沈予安，是沈予岚。', 'clarify', '沈予岚的哪项'),
    ('饮品当前版。', 'read', 7),
    ('请查饮品。', 'clarify', '谁的饮品'),
    ('沈予安。', 'clarify', '饮品哪一版'),
    ('首版。', 'read', 1),
    ('不需要翻档。讨论一下怎样给饮品分类。', 'chat', tuple(VALUES.values())),
    ('请查用品当前版。', 'clarify', '谁的用品'),
    ('先别查了。', 'defer', None),
    ('沈予安。', 'chat', None),
    ('请查沈予安的首版。', 'clarify', '哪项记录'),
    ('查沈予岚的饮品。', 'clarify', '饮品哪一版'),
    ('当前版。', 'read', 7),
    ('请查饮品上一版。', 'clarify', '谁的饮品'),
)
RESTART = (
    ('沈予安。', 'chat', None),
    ('请查谁的饮品上一版。', 'clarify', '谁的饮品'),
    ('是沈予安。', 'read', 1),
    ('请查见面地点当前版。', 'clarify', '谁的见面地点'),
    ('换个话题，解释一下什么是回声。', 'chat', None),
    ('沈予安。', 'chat', None),
    ('请查谁的用品当前版。', 'clarify', '谁的用品'),
    ('沈予安还是沈予岚？', 'clarify', '姓名'),
    ('沈予安。', 'read', 5),
    ('请查用品当前版。', 'clarify', '谁的用品'),
    ('/清空对话', 'reset', None),
    ('沈予安。', 'chat', None),
    ('请查沈予安的见面地点当前版。', 'read', 4),
)


def replay(profile, cases, recurrent):
    start = perf_counter()
    command = ['cmd.exe', '/d', '/c', 'spora_version_chat.cmd', '--profile', str(profile),
               '--reply-tokens', '32', '--json']
    if recurrent:
        command.extend(('--recurrent', *map(str, recurrent)))
    process = subprocess.run(command, cwd=ROOT, input='\n'.join([*(row[0] for row in cases), '/退出', '']),
                             encoding='utf-8', capture_output=True, timeout=270)
    if process.returncode:
        raise RuntimeError(process.stderr)
    responses = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if len(responses) != len(cases):
        raise RuntimeError(f'Expected {len(cases)} replies, got {len(responses)}. {process.stderr.strip()}')
    rows = []
    for (question, kind, expectation), response in zip(cases, responses, strict=True):
        answer = response['answer']
        selected = response.get('selected', [])
        passed = response['kind'] == kind
        if kind == 'read':
            value = VALUES[expectation]
            passed &= selected == [expectation] and answer.count(value) == 1
            passed &= not any(other in answer for other in VALUES.values() if other != value)
        elif kind == 'saved':
            passed &= response.get('record') == expectation
        else:
            passed &= not selected
            if kind == 'clarify':
                passed &= expectation in answer and not any(value in answer for value in VALUES.values())
            if kind == 'defer':
                passed &= not any(value in answer for value in VALUES.values())
            if kind == 'chat' and expectation:
                passed &= not any(value in answer for value in expectation)
        rows.append({'question': question, 'kind': kind, 'expected': expectation, 'actual': response, 'passed': passed})
        if not passed:
            print(rows[-1], flush=True)
    return {'rows': rows, 'passed': sum(row['passed'] for row in rows), 'seconds': perf_counter() - start}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    profile = (args.output / 'profile').resolve()
    first = replay(profile, FIRST, args.recurrent)
    database = profile / 'versions.sqlite3'
    before = sha256(database.read_bytes()).hexdigest()
    restart = replay(profile, RESTART, args.recurrent)
    after = sha256(database.read_bytes()).hexdigest()
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as conn:
        verified = conn.execute('pragma integrity_check').fetchone() == ('ok',)
        count, = conn.execute('select count(*) from versions').fetchone()
    archive_ok = verified and count == len(VALUES) and before == after
    store_json(args.output / 'result.json', {
        'first': first, 'restart': restart, 'archive_ok': archive_ok, 'archive_sha256': after, 'record_count': count,
    })
    print(f"first {first['passed']}/{len(FIRST)}, restart {restart['passed']}/{len(RESTART)}, archive {archive_ok}",
          flush=True)
    if first['passed'] != len(FIRST) or restart['passed'] != len(RESTART) or not archive_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--recurrent', type=Path, nargs=3, metavar=('CHECKPOINT', 'LAYER19', 'LAYER22'))
    main(parser.parse_args())
