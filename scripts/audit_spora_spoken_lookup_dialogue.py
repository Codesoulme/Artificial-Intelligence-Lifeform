"""Replay user confirmation, correction, and scope expiry in the Windows entry."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from scripts.audit_spora_natural_lookup_cli import digest, replay
from scripts.audit_spora_spoken_temporal_head import RECORDS
from scripts.train_spora_recurrent_counterfactual import store_json


PLACE = '罗映禾的约见地，最后一次挪地方之前在哪儿？'
ITEM = '穆云川要带的东西是哪件？'
CONFIRM_PLACE = ('clarify', '罗映禾的见面地点', 'unconfirmed_address')
CONFIRM_ITEM = ('clarify', '穆云川的用品', 'unconfirmed_address')
RESET = ('/清空对话', 'reset', None, None)

FIRST = (
    (PLACE, *CONFIRM_PLACE),
    ('上一版。', *CONFIRM_PLACE),
    ('是。', 'clarify', '罗映禾的见面地点哪一版', 'incomplete_address'),
    ('上一版。', 'read', 5, 'located'),
    RESET,
    (ITEM, *CONFIRM_ITEM),
    ('是，上一版。', 'read', 8, 'located'),
    RESET,
    (PLACE, *CONFIRM_PLACE),
    ('不是罗映禾的见面地点，是罗映荷的饮品首版。', 'read', 1, 'located'),
    RESET,
    (PLACE, *CONFIRM_PLACE),
    ('不对。', 'clarify', '重新告诉我', 'address_declined'),
    ('是，上一版。', 'chat', None, None),
    RESET,
    (ITEM, *CONFIRM_ITEM),
    ('先别查了。', 'defer', None, None),
    ('是，上一版。', 'chat', None, None),
    RESET,
    (ITEM, *CONFIRM_ITEM),
    ('换个话题，海星有几条腕？', 'chat', None, None),
    ('是，首版。', 'chat', None, None),
    RESET,
    (PLACE, *CONFIRM_PLACE),
    ('把「是，上一版」翻译成英文。', 'chat', None, None),
    ('是，上一版。', 'chat', None, None),
    RESET,
    (PLACE, *CONFIRM_PLACE),
    RESET,
    ('是，上一版。', 'chat', None, None),
    (ITEM, *CONFIRM_ITEM),
)
RESTART = (
    ('是，上一版。', 'chat', None, None),
    (PLACE, *CONFIRM_PLACE),
    ('对，首版还是当前版？', 'clarify', '哪一版', 'incomplete_address'),
    ('首版。', 'read', 4, 'located'),
    RESET,
    ('请查穆云川的用品当前版。', 'read', 9, 'located'),
)
FRESH = (
    ('我想知道这趟要到哪儿跟罗映禾见面。后来重选的地方在哪？', '罗映禾的见面地点'),
    ('穆云川让我拿点东西。刚提过要拿的是哪一个？', '穆云川的用品'),
    ('罗映荷老早挑过一杯喝的，头一回挑的是什么？', '罗映荷的饮品'),
    ('要去给叶盼山取书，先前换掉的那一本叫什么？', '叶盼山的书目'),
    ('罗映荷先不用问，罗映禾约我在哪儿见面？用现在有效的地点。', '罗映禾的见面地点'),
)


def read_only(path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)


def compare(expected, replies):
    values = [value for _, _, value in RECORDS]
    rows = []
    for (question, kind, wanted, reason), observed in zip(expected, replies, strict=True):
        answer = observed['answer']
        selections = [wanted] if kind == 'read' else []
        shown = [value for value in values if value in answer]
        passed = (observed['kind'] == kind and observed.get('selected', []) == selections
                  and shown == ([values[wanted - 1]] if kind == 'read' else [])
                  and (reason is None or reason in observed.get('reasons', ()))
                  and (kind != 'clarify' or wanted in answer))
        row = dict(question=question, kind=kind, wanted=wanted, reason=reason, observed=observed, passed=passed)
        if not passed:
            print('check', row, flush=True)
        rows.append(row)
    return rows


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    profile = args.output.resolve() / 'profile'
    profile.mkdir()
    db = profile / 'versions.sqlite3'
    source = args.source / 'profile' / 'versions.sqlite3'
    sha = digest(source)
    source_report = json.loads((args.source / 'result.json').read_text(encoding='utf-8'))
    if sha != next(value for path, value in source_report['sources'].items()
                   if Path(path).resolve() == source.resolve()):
        raise ValueError('The sealed three-version profile changed')
    with closing(read_only(source)) as connection, closing(sqlite3.connect(db)) as copied:
        connection.backup(copied)
    options = ('--spoken-lookup', args.planner)
    initial, first_seconds = replay(profile, [line for line, *_ in FIRST], args.recurrent, extra_options=options)
    first = compare(FIRST, initial)
    sample = [line for question, _ in FRESH for line in ('/清空对话', question)]
    replies, restart_seconds = replay(profile, [line for line, *_ in RESTART] + sample, args.recurrent,
                                       extra_options=options)
    restart = compare(RESTART, replies[:len(RESTART)])
    fresh = [dict(question=question, expected=expected, response=response,
                  proposed=expected in response['answer'] and 'unconfirmed_address' in response.get('reasons', []))
             for (question, expected), response in zip(FRESH, replies[len(RESTART) + 1::2], strict=True)]
    with closing(read_only(db)) as check, closing(read_only(source)) as original:
        checked = (check.execute('pragma integrity_check').fetchone() == ('ok',)
                   and check.execute('select * from versions order by number').fetchall()
                   == original.execute('select * from versions order by number').fetchall())
    if digest(source) != sha or not checked:
        raise ValueError('The replay changed the source memory or its isolated copy')
    unexpected = [row for row in fresh if row['response'].get('selected')]
    store_json(args.output / 'result.json', dict(first=first, restart=restart, fresh=fresh, source_sha256=sha,
               archive_ok=checked, first_seconds=first_seconds, restart_seconds=restart_seconds))
    print(f"first {sum(row['passed'] for row in first)}/{len(first)}, restart {sum(row['passed'] for row in restart)}/{len(restart)}, "
          f"fresh proposals {sum(row['proposed'] for row in fresh)}/{len(fresh)}", flush=True)
    if not all(row['passed'] for row in (*first, *restart)) or unexpected:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('planner', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
