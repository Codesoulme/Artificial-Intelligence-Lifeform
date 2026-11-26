"""Replay lookup cancellations through the local Windows chat entry point."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from scripts.audit_spora_natural_lookup_cli import digest, replay
from scripts.audit_spora_spoken_lookup_dialogue import read_only
from scripts.audit_spora_spoken_switch_cli import matching_read, no_read
from scripts.train_spora_recurrent_counterfactual import store_json


OPEN = '请查岑蕙安的书目当前版。'
REOPEN = '请查岑蕙安的书目上一版。'
FOLLOWUP = '上一版。'
SCENARIOS = (
    ('confirmed', OPEN, '先不查了。', 'defer'),
    ('pending', '请查岑蕙安的书目。', '先别查了。', 'defer'),
    ('pair', '请查岑蕙安的书目；查唐砚平的见面地点首版。', '先暂停。', 'defer'),
    ('negation', OPEN, '本次调档无须接着办。', 'defer'),
    ('colloquial', OPEN, '这轮查阅到此为止。', 'defer'),
    ('cancel_pending_with_new_wording', '请查岑蕙安的书目。', '本轮调阅就此结束！', 'defer'),
    ('quotation', OPEN, '把「这轮查阅到此为止」翻译成英文。', 'chat'),
    ('denial', OPEN, '这轮查阅不要停止。', 'chat'),
    ('new_task', OPEN, '先不查了，帮我想一个整理书桌的方法。', 'chat'),
    ('continue', OPEN, '再查刚才查到的那项上一版。', 'read'),
)


def main(args):
    source = args.archive / 'three-versions.sqlite3'
    manifest = json.loads((args.archive / 'result.json').read_text(encoding='utf-8'))
    values = {int(number): value for number, value in manifest['values'].items()}
    before = digest(source)
    if before != manifest['archive_sha256']:
        raise ValueError('The archive differs from its report')
    profile = (args.output / 'profile').resolve()
    profile.mkdir(parents=True, exist_ok=False)
    database = profile / 'versions.sqlite3'
    with closing(read_only(source)) as original, closing(sqlite3.connect(database)) as copy:
        original.backup(copy)
    prompts = [line for _, opening, action, _ in SCENARIOS
               for line in ('/清空对话', opening, action, FOLLOWUP, REOPEN)]
    replies, seconds = replay(profile, prompts, args.recurrent, extra_options=('--spoken-lookup', args.planner))
    rows = []
    for index, (name, opening_text, action, kind) in enumerate(SCENARIOS):
        reset, opening, acted, followup, reopened = replies[index * 5:index * 5 + 5]
        expected_open = (matching_read(opening, 3, values) if opening_text == OPEN
                         else opening['kind'] == 'clarify' and no_read(opening, values))
        if reset['kind'] != 'reset' or not expected_open:
            raise ValueError(f'Failed to establish lookup scope for {name}: {opening}')
        if kind == 'read':
            action_ok = matching_read(acted, 2, values)
            followup_ok = matching_read(followup, 2, values)
        else:
            action_ok = acted['kind'] == kind and no_read(acted, values)
            followup_ok = (followup['kind'] == 'clarify' and no_read(followup, values)
                           and 'incomplete_address' in followup['reasons'])
        rows.append(dict(name=name, action=action, expected_kind=kind, opening=opening, reply=acted,
                         followup=followup, reopened=reopened, action_ok=action_ok, followup_ok=followup_ok,
                         reopened_ok=matching_read(reopened, 2, values)))
    with closing(read_only(source)) as original, closing(read_only(database)) as copy:
        query = 'select * from versions order by number'
        intact = (copy.execute('pragma integrity_check').fetchone() == ('ok',)
                  and copy.execute(query).fetchall() == original.execute(query).fetchall())
    if not intact or digest(source) != before:
        raise ValueError('The replay changed the source or its isolated archive')
    summary = {key: sum(row[key] for row in rows) for key in ('action_ok', 'followup_ok', 'reopened_ok')}
    summary.update(total=len(rows), turns=len(prompts))
    store_json(args.output / 'result.json', dict(rows=rows, summary=summary, seconds=seconds, archive_sha256=before))
    print(summary, flush=True)
    if any(summary[key] != len(rows) for key in ('action_ok', 'followup_ok', 'reopened_ok')):
        raise SystemExit(1)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('archive', type=Path)
    cli.add_argument('planner', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(cli.parse_args())
