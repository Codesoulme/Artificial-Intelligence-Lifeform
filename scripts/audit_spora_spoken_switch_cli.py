"""Replay natural switches from a confirmed address with a sealed version archive."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from scripts.audit_spora_natural_lookup_cli import digest, replay
from scripts.audit_spora_spoken_lookup_dialogue import read_only
from scripts.train_spora_recurrent_counterfactual import store_json


OPEN = '请查岑蕙安的书目当前版。'
SCENARIOS = (
    ('location', '改查唐砚平约我在哪儿碰头，后来换的地方在哪？', '唐砚平的见面地点',
     '请查唐砚平的见面地点上一版。', 5),
    ('item', '现在换查叶澄依让我拿的那个东西叫什么。', '叶澄依的用品', '请查叶澄依的用品首版。', 7),
    ('drink', '那就改查顾遥青后来改喝哪种饮料。', '顾遥青的饮品', '请查顾遥青的饮品当前版。', 12),
    ('unknown', '改查闻锦潮借的那一本。', None, None, None),
    ('alternatives', '换查唐砚平的见面地点或者顾遥青的饮品。', None, None, None),
    ('literal', '改查顾遥青的书目上一版。', None, None, 14),
    ('missing_version', '换查唐砚平的见面地点。', None, None, 5),
)


def matching_read(reply, number, values):
    return (reply['kind'] == 'read' and reply.get('selected') == [number]
            and {value for value in values.values() if value in reply['answer']} == {values[number]})


def no_read(reply, values):
    return not reply.get('selected') and not any(value in reply['answer'] for value in values.values())


def main(args):
    source = args.archive / 'three-versions.sqlite3'
    report = json.loads((args.archive / 'result.json').read_text(encoding='utf-8'))
    values = {int(number): value for number, value in report['values'].items()}
    before = digest(source)
    if before != report['archive_sha256']:
        raise ValueError('The archive differs from its report')
    profile = (args.output / 'profile').resolve()
    profile.mkdir(parents=True, exist_ok=False)
    db = profile / 'versions.sqlite3'
    with closing(read_only(source)) as connection, closing(sqlite3.connect(db)) as copied:
        connection.backup(copied)

    steps = []
    for name, prompt, address, explicit, number in SCENARIOS:
        steps.extend(('/清空对话', OPEN, prompt))
        if address:
            steps.extend(('上一版。', '是，上一版。' if number == 5 else
                          '是，首版。' if number == 7 else '是，当前版。', '先不查了。', explicit))
        elif name in ('unknown', 'alternatives'):
            steps.extend(('首版。', '先不查了。', '首版。'))
        elif name == 'missing_version':
            steps.append('上一版。')
    replies, seconds = replay(profile, steps, args.recurrent, extra_options=('--spoken-lookup', args.planner))
    seen = iter(replies)
    rows = []
    for name, prompt, address, _, number in SCENARIOS:
        reset, opening, proposal = next(seen), next(seen), next(seen)
        if reset['kind'] != 'reset' or not matching_read(opening, 3, values):
            raise ValueError(f'Failed to establish the confirmed address in {name}: {opening}')
        row = dict(case=name, question=prompt, proposal=proposal)
        if address:
            bare, confirm, cancel, explicit = (next(seen) for _ in range(4))
            proposed = address in proposal['answer'] and 'unconfirmed_address' in proposal.get('reasons', ())
            details = ('unconfirmed_address' in bare.get('reasons', ()) and matching_read(confirm, number, values))
            row.update(proposed=proposed, confirmed=details, bare=bare, confirm=confirm, cancel=cancel, explicit=explicit,
                       passed=no_read(proposal, values) and proposal['kind'] == 'clarify' and no_read(bare, values)
                       and (not confirm.get('selected') or matching_read(confirm, number, values))
                       and no_read(cancel, values) and matching_read(explicit, number, values))
        elif name == 'literal':
            row['passed'] = matching_read(proposal, number, values)
        elif name == 'missing_version':
            followup = next(seen)
            row.update(followup=followup, passed=proposal['kind'] == 'clarify' and no_read(proposal, values)
                       and '唐砚平的见面地点哪一版' in proposal['answer'] and matching_read(followup, number, values))
        else:
            bare, cancel, after = (next(seen) for _ in range(3))
            row.update(bare=bare, cancel=cancel, after=after,
                       passed=all(item['kind'] == 'clarify' and no_read(item, values) for item in (proposal, bare, after))
                       and no_read(cancel, values))
        rows.append(row)
    with closing(read_only(db)) as check, closing(read_only(source)) as original:
        query = 'select * from versions order by number'
        intact = (check.execute('pragma integrity_check').fetchone() == ('ok',)
                  and check.execute(query).fetchall() == original.execute(query).fetchall())
    if not intact or digest(source) != before:
        raise ValueError('The replay changed the source or its isolated archive')
    summary = dict(passed=sum(row['passed'] for row in rows), total=len(rows),
                   proposed=sum(row.get('proposed', False) for row in rows),
                   confirmed=sum(row.get('confirmed', False) for row in rows), turns=len(steps))
    store_json(args.output / 'result.json', dict(rows=rows, summary=summary, seconds=seconds, archive_sha256=before))
    print(summary, flush=True)
    if summary['passed'] != summary['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('archive', type=Path)
    cli.add_argument('planner', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(cli.parse_args())
