"""Exercise the Windows entry point through writes, restart and composed replies."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess
from time import perf_counter

from scripts.train_spora_recurrent_counterfactual import store_json


ROOT = Path(__file__).resolve().parents[1]
RECORDS = (
    ('许林秋', '见面地点', '白榆亭'),
    ('许林秋', '见面地点', '杏叶廊'),
    ('许林秋', '用品', '青布袋'),
    ('许林秋', '用品', '灰毛帽'),
    ('姚清婉', '书目', '河口雨声'),
    ('姚清婉', '书目', '烟岛旧书'),
    ('姚清婉', '见面地点', '青砖厅'),
    ('姚清婉', '见面地点', '南帆馆'),
    ('冯远汐', '书目', '老街回响'),
    ('冯远汐', '书目', '雨后山色'),
    ('冯远汐', '饮品', '蜜桃茶'),
    ('冯远汐', '饮品', '杏仁乳'),
)
MEETING = '给许林秋写一句能直接发的微信，说明周五傍晚在哪里碰面、需要带什么。只给消息正文。'
BOOK = '给姚清婉写一句短消息，请她今晚七点带已登记的那本书到约好的地方见。直接给能发送的一句话。'
DRINK = '给冯远汐发一句话，告诉她下午借的书已经准备好了，顺便说会给她带喜欢的饮品。把书名和饮品名写进消息。'
CURRENT = '/成句 许林秋 | 见面地点当前版 | 用品当前版 | 周五傍晚 | ' + MEETING
MIXED = '/成句 许林秋 | 见面地点当前版 | 用品上一版 | 周五傍晚 | ' + MEETING
CHANGED = CURRENT.replace('周五傍晚', '星期三中午')
REORDERED = CURRENT.replace('见面地点当前版 | 用品当前版', '用品当前版 | 见面地点当前版')
WRITES = tuple((f'/记 {name} | {field} | {value}', 'saved', (i,))
               for i, (name, field, value) in enumerate(RECORDS, 1))
FIRST = (*WRITES, (CURRENT, 'composed', (2, 4)), (MIXED, 'composed', (2, 3)))
RESTART = (
    ('那上一版呢？', 'clarify', ()),
    (CURRENT, 'composed', (2, 4)),
    (MIXED, 'composed', (2, 3)),
    (CHANGED, 'composed', (2, 4)),
    (REORDERED, 'composed', (4, 2)),
    ('/成句 姚清婉 | 书目首版 | 见面地点当前版 | 今晚七点 | ' + BOOK, 'composed', (5, 8)),
    ('/成句 冯远汐 | 饮品上一版 | 书目当前版 | 下午 | ' + DRINK, 'composed', (11, 10)),
    ('/成句 许林秋 | 见面地点第九版 | 用品当前版 | 周五傍晚 | ' + MEETING, 'clarify', ()),
    ('/成句 邹嘉齐 | 见面地点当前版 | 用品当前版 | 周五傍晚 | ' + MEETING, 'clarify', ()),
    ('请查许林秋的用品首版。', 'read', (3,)),
    (CURRENT.replace('见面地点当前版', '见面地点首版还是当前版'), 'error', ()),
    ('/清空对话', 'reset', ()),
    (CURRENT, 'composed', (2, 4)),
)


def replay(profile, cases, composition):
    start = perf_counter()
    process = subprocess.run(
        ['cmd.exe', '/d', '/c', 'spora_version_chat.cmd', '--profile', str(profile),
         '--composition', str(composition), '--json'],
        cwd=ROOT, input='\n'.join([*(question for question, _, _ in cases), '/退出', '']),
        encoding='utf-8', capture_output=True, timeout=270,
    )
    if process.returncode:
        raise RuntimeError(process.stderr)
    results = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    checks = []
    for (question, kind, numbers), output in zip(cases, results, strict=True):
        selected = (output['record'],) if output['kind'] == 'saved' else tuple(output.get('selected', ()))
        answer = output['answer']
        valid = output['kind'] == kind and numbers == selected
        if kind in ('composed', 'read'):
            expected = {RECORDS[number - 1][2] for number in numbers}
            present = {value for _, _, value in RECORDS if value in answer}
            valid &= expected == present and all(answer.count(value) == 1 for value in expected)
        if kind == 'composed':
            when = question.split('|', 4)[3].strip()
            valid &= answer.count(when) == 1 and answer[-1:] in '。！？.!?'
            valid &= len(output.get('copied', ())) == 3
        checks.append({'question': question, 'result': output, 'expected_kind': kind,
                       'expected_records': numbers, 'passed': valid})
        if not valid:
            print(checks[-1], flush=True)
    return {'checks': checks, 'seconds': perf_counter() - start, 'passed': sum(row['passed'] for row in checks)}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    profile = (args.output / 'profile').resolve()
    first = replay(profile, FIRST, args.composition)
    database = profile / 'versions.sqlite3'
    before = sha256(database.read_bytes()).hexdigest()
    restart = replay(profile, RESTART, args.composition)
    after = sha256(database.read_bytes()).hexdigest()
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        intact = connection.execute('pragma integrity_check').fetchone() == ('ok',)
        count, = connection.execute('select count(*) from versions').fetchone()
    current, mixed = [row['result'] for row in first['checks'][-2:]]
    replayed = [row['result'] for row in restart['checks']]
    answer = current['answer']
    composition_checks = {
        'same_after_restart': current == replayed[1] and mixed == replayed[2],
        'only_version_value_changes': mixed['answer'] == answer.replace('灰毛帽', '青布袋'),
        'only_time_changes': replayed[3]['answer'] == answer.replace('周五傍晚', '星期三中午'),
        'record_order_preserves_sentence': replayed[4]['answer'] == answer,
        'same_after_clear': replayed[-1] == current,
        'archive_intact': intact and count == len(RECORDS) and before == after,
    }
    store_json(args.output / 'result.json', {'first': first, 'restart': restart, 'controls': composition_checks,
                                              'records': count, 'database_sha256': after})
    print(f"first {first['passed']}/{len(FIRST)}, restart {restart['passed']}/{len(RESTART)}",
          composition_checks, flush=True)
    if not all(composition_checks.values()) or first['passed'] != len(FIRST) or restart['passed'] != len(RESTART):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--composition', type=Path, default=Path('runs/spora-typed-composition-v1'))
    main(parser.parse_args())
