"""Replay reuse, replacement and expiry of confirmed addresses in the Windows CLI."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from scripts.audit_spora_natural_lookup_cli import digest, replay
from scripts.audit_spora_spoken_lookup_dialogue import read_only
from scripts.audit_spora_spoken_temporal_head import RECORDS
from scripts.train_spora_recurrent_counterfactual import store_json


REFERENCE = '刚才核对过的那份记录，麻烦再查一下上一版。'
UNKNOWN = ('clarify', (), 'incomplete_address')
FIRST = (
    ('请查罗映禾的见面地点当前版。', 'read', (6,), 'located'),
    (REFERENCE, 'read', (5,), 'located'),
    ('请再核对刚刚查到的那项。', 'clarify', (), 'unresolved_scope'),
    ('首版。', 'read', (4,), 'located'),
    ('换个话题，帮我说一句早安。', 'chat', (), None),
    (REFERENCE, *UNKNOWN),
    ('请查罗映禾的见面地点当前版。', 'read', (6,), 'located'),
    ('请查罗映荷的饮品当前版。', 'read', (3,), 'located'),
    ('请再查刚才确认过的那条首版。', 'read', (1,), 'located'),
    ('先不查了。', 'defer', (), None),
    (REFERENCE, *UNKNOWN),
    ('请查罗映禾的见面地点当前版；查罗映荷的饮品当前版。', 'read', (6, 3), 'located'),
    (REFERENCE, *UNKNOWN),
    ('请查穆云川的用品当前版。', 'read', (9,), 'located'),
    ('把刚才查到的那项写成一句话。', 'chat', (), None),
    (REFERENCE, *UNKNOWN),
    ('请查穆云川的用品当前版。', 'read', (9,), 'located'),
    ('/清空对话', 'reset', (), None),
    (REFERENCE, *UNKNOWN),
    ('请查穆云川的用品当前版。', 'read', (9,), 'located'),
)
RESTART = (
    (REFERENCE, *UNKNOWN),
    ('穆云川的用品。', 'read', (8,), 'located'),
    ('请查叶盼山的书目当前版。', 'read', (12,), 'located'),
    (REFERENCE, 'read', (11,), 'located'),
)


def check(cases, replies):
    values = {index: value for index, (_, _, value) in enumerate(RECORDS, 1)}
    rows = []
    for (text, kind, records, reason), reply in zip(cases, replies, strict=True):
        selected = tuple(reply.get('selected', []))
        shown = {index for index, value in values.items() if value in reply['answer']}
        passed = (reply['kind'] == kind and selected == records
                  and (reason is None or reason in reply.get('reasons', ()))
                  and (reason != 'incomplete_address' or ('谁的哪项记录' in reply['answer'] and not shown))
                  and (not records or shown == set(records)))
        rows.append(dict(question=text, expected_kind=kind, expected_records=records, reply=reply, passed=passed))
    return rows


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    profile = (args.output / 'profile').resolve()
    profile.mkdir()
    source = args.source / 'profile' / 'versions.sqlite3'
    history = json.loads((args.source / 'result.json').read_text(encoding='utf-8'))
    checksum = digest(source)
    expected = {Path(path).resolve(): value for path, value in history['sources'].items()}
    if checksum != expected.get(source.resolve()):
        raise ValueError('Expected the sealed three-version profile')
    archive = profile / source.name
    with closing(read_only(source)) as original, closing(sqlite3.connect(archive)) as copy:
        original.backup(copy)
    first, first_seconds = replay(profile, [text for text, *_ in FIRST], args.recurrent)
    restart, restart_seconds = replay(profile, [text for text, *_ in RESTART], args.recurrent)
    checks = {'first': check(FIRST, first), 'restart': check(RESTART, restart)}
    with closing(read_only(archive)) as db, closing(read_only(source)) as original:
        integrity, = db.execute('PRAGMA integrity_check').fetchone()
        records = 'SELECT * FROM versions ORDER BY number'
        identical = db.execute(records).fetchall() == original.execute(records).fetchall()
    intact = integrity == 'ok' and identical and digest(source) == checksum
    store_json(args.output / 'result.json', dict(checks=checks, archive_ok=intact,
               archive_sha256=checksum, first_seconds=first_seconds, restart_seconds=restart_seconds))
    for stage, rows in checks.items():
        failures = [row for row in rows if not row['passed']]
        print(stage, f'{len(rows) - len(failures)}/{len(rows)}', failures, flush=True)
    if not intact or any(not row['passed'] for rows in checks.values() for row in rows):
        raise RuntimeError('Confirmed address replay failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
