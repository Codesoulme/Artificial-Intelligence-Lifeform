"""Replay paired version requests and clarifications against a fresh local archive."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess
from time import perf_counter

from scripts.audit_spora_version_local_chat import ROOT
from scripts.train_spora_recurrent_counterfactual import store_json


RECORDS = (
    ('岑蕙安', '书目', '春潮', '远山'),
    ('唐砚平', '见面地点', '南风巷', '云桥'),
    ('叶澄依', '用品', '纸风车', '毛毡笔袋'),
    ('顾遥青', '饮品', '桂花露', '盐汽水'),
    ('顾遥青', '书目', '灯塔', '白鲸'),
)
VALUES = {number: value for number, value in enumerate(
    (value for _, _, first, current in RECORDS for value in (first, current)), 1)}
WRITES = tuple((f'/记 {owner} | {field} | {value}', 'saved', number)
               for index, (owner, field, first, current) in enumerate(RECORDS)
               for number, value in ((2 * index + 1, first), (2 * index + 2, current)))

CASES = {
    'complete': (
        (('请查岑蕙安的书目首版；再查唐砚平的见面地点当前版。', 'read', (1, 4)),),
        (('核对叶澄依的用品当前版；查顾遥青的饮品首版。', 'read', (6, 7)),),
        (('请查顾遥青的书目当前版；核对岑蕙安的书目首版。', 'read', (10, 1)),),
    ),
    'missing': (
        (('请查岑蕙安的书目首版；再核对唐砚平的见面地点。', 'clarify', '唐砚平的见面地点'),
         ('上一版。', 'read', (1, 3))),
        (('查岑蕙安的书目；查唐砚平的见面地点上一版。', 'clarify', '岑蕙安的书目'),
         ('当前版。', 'read', (2, 3))),
        (('请核对叶澄依的用品；请查顾遥青的饮品。', 'clarify', '叶澄依的用品'),
         ('首版。', 'clarify', '顾遥青的饮品'),
         ('当前版。', 'read', (5, 8))),
        (('查顾遥青的书目；查岑蕙安的书目。', 'clarify', '顾遥青的书目'),
         ('首版还是当前版？', 'clarify', '顾遥青的书目'),
         ('顾遥青的书目首版。', 'clarify', '岑蕙安的书目'),
         ('书目当前版。', 'read', (9, 2))),
        (('核对岑蕙安的书目；再查叶澄依的用品。', 'clarify', '岑蕙安的书目'),
         ('两项都查首版。', 'read', (1, 5))),
    ),
    'no_read': (
        (('不需要翻档。帮我想想怎么布置读书会。', 'chat', None),),
        (('把「查唐砚平的见面地点首版；查岑蕙安的书目首版」改得简短点。', 'chat', None),),
        (('核对顾遥青的饮品；查叶澄依的用品当前版。', 'clarify', '顾遥青的饮品'),
         ('先不查了。', 'defer', None),
         ('首版。', 'chat', None)),
        (('核对顾遥青的书目；查叶澄依的用品当前版。', 'clarify', '顾遥青的书目'),
         ('请查唐砚平的见面地点当前版。', 'read', (4,)),
         ('首版。', 'chat', None)),
    ),
}
PREFLIGHT = ('/清空对话', 'reset', None)
BEFORE_EXIT = ('请查唐砚平的见面地点；查岑蕙安的书目首版。', 'clarify', '唐砚平的见面地点')
RESTART = (
    ('首版。', 'chat', None),
    ('请查唐砚平的见面地点首版；查岑蕙安的书目当前版。', 'read', (3, 2)),
)


def run(profile, steps, recurrent):
    command = ['cmd.exe', '/d', '/c', 'spora_version_chat.cmd', '--profile', str(profile),
               '--reply-tokens', '24', '--json']
    if recurrent:
        command.extend(('--recurrent', *map(str, recurrent)))
    lines = '\n'.join([*(question for question, _, _ in steps), '/退出', ''])
    process = subprocess.run(command, cwd=ROOT, input=lines, encoding='utf-8',
                             capture_output=True, check=True, timeout=270)
    replies = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if len(replies) != len(steps):
        raise RuntimeError(f'Expected {len(steps)} replies, got {len(replies)}. {process.stderr.strip()}')
    return replies


def grade(steps, replies):
    rows = []
    for (question, kind, expected), result in zip(steps, replies, strict=True):
        actual = tuple(result.get('selected', ()))
        answer = result['answer']
        observed = {value for value in VALUES.values() if value in answer}
        passed = result['kind'] == kind
        if kind == 'saved':
            passed &= result.get('record') == expected
        elif kind == 'read':
            passed &= actual == expected and observed == {VALUES[number] for number in expected}
        else:
            passed &= not actual and not observed and (kind != 'clarify' or expected in answer)
        rows.append(dict(question=question, kind=kind, expected=expected, result=result, passed=passed))
    return rows


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    profile = (args.output / 'profile').resolve()
    dialogues = [cases for group in CASES.values() for cases in group]
    steps = (*WRITES, *(step for cases in dialogues for step in (PREFLIGHT, *cases)), PREFLIGHT, BEFORE_EXIT)
    rows = grade(steps, run(profile, steps, args.recurrent))
    archive = profile / 'versions.sqlite3'
    before = sha256(archive.read_bytes()).hexdigest()
    restart = grade(RESTART, run(profile, RESTART, args.recurrent))
    after = sha256(archive.read_bytes()).hexdigest()
    with sqlite3.connect(archive.as_uri() + '?mode=ro', uri=True) as connection:
        archive_ok = connection.execute('pragma integrity_check').fetchone() == ('ok',)
        count, = connection.execute('select count(*) from versions').fetchone()
    archive_ok &= count == len(VALUES) and before == after and all(row['passed'] for row in rows[:len(WRITES)])
    index = len(WRITES)
    summary = {}
    for group, cases in CASES.items():
        scores = []
        for dialogue in cases:
            end = index + 1 + len(dialogue)
            scores.append(all(row['passed'] for row in rows[index:end]))
            index = end
        summary[group] = (sum(scores), len(scores))
    summary['restart'] = (sum(row['passed'] for row in restart), len(restart))
    for row in (*rows, *restart):
        if not row['passed']:
            print(row, flush=True)
    print(summary, 'archive', archive_ok, flush=True)
    store_json(args.output / 'result.json', dict(rows=rows, restart=restart, summary=summary,
               archive_ok=archive_ok, archive_sha256=after, seconds=perf_counter() - started))
    if not archive_ok or (args.require_all and not all(row['passed'] for row in (*rows, *restart))):
        raise SystemExit(1)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, metavar=('CHECKPOINT', 'LAYER19', 'LAYER22'))
    cli.add_argument('--require-all', action='store_true')
    main(cli.parse_args())
