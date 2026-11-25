"""Test natural memory address selection against explicit and no-read controls."""

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
from time import perf_counter

from scripts.audit_spora_fresh_dialogue import digest
from scripts.audit_spora_version_local_chat import ROOT
from scripts.train_spora_recurrent_counterfactual import store_json


RECORDS = (
    ('陶依萍', '饮品', '麦芽茶'), ('陶依萍', '饮品', '梅子汤'),
    ('陶依萍', '见面地点', '芦苇亭'), ('陶依萍', '见面地点', '木桥边'),
    ('陶依晴', '饮品', '杏仁奶'), ('陶依晴', '用品', '花布袋'),
    ('许雁书', '书目', '山城信'), ('许雁书', '书目', '石阶雨'),
)

# The paired prompts use the same person and field vocabulary. A name or field
# being mentioned is insufficient when the user asks to translate or not read.
PAIRS = (
    (2, '请查陶依萍的饮品当前版。',
     '陶依萍最近改喝什么了？我买前想确认一下。',
     '帮我润色这句，不用回答里面的问题：陶依萍最近改喝什么了？'),
    (1, '请查陶依萍的饮品上一版。',
     '陶依萍换饮料以前，选的是什么？',
     '陶依萍换饮料以前选的是什么，这句能不能译成英文？'),
    (4, '请查陶依萍的见面地点当前版。',
     '我得去和陶依萍碰头。后来改到哪儿见来着？',
     '我得去和陶依萍碰头。先别查地点，帮我问她会晚到几分钟。'),
    (5, '请查陶依晴的饮品当前版。',
     '陶依萍的先不看。陶依晴那杯最后定的是什么？',
     '先不用查陶依萍和陶依晴喝什么，帮我写一句问两人到了没有。'),
    (6, '请查陶依晴的用品当前版。',
     '给陶依晴准备东西时，她原来让我捎的那件是什么？',
     '给陶依晴准备东西时，她原来让我捎的那件是什么？把这句话改得礼貌些。'),
    (7, '请查许雁书的书目首版。',
     '许雁书一开始选的是哪本书？我想找最早那一本。',
     '许雁书一开始选的是哪本书？请照抄这句，别查书名。'),
)


def replay(profile, questions, recurrent, *, extra_options=()):
    command = ['cmd.exe', '/d', '/c', 'spora_version_chat.cmd', '--profile', str(profile),
               '--reply-tokens', '72', '--json']
    if recurrent is not None:
        command.extend(('--recurrent', *map(str, recurrent)))
    command.extend(map(str, extra_options))
    started = perf_counter()
    process = subprocess.run(command, cwd=ROOT, input='\n'.join([*questions, '/退出', '']),
                             encoding='utf-8', capture_output=True, timeout=360)
    if process.returncode:
        raise RuntimeError(process.stderr)
    answers = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if len(answers) != len(questions):
        raise ValueError(f'Expected {len(questions)} replies, received {len(answers)}')
    return answers, perf_counter() - started


def evaluate(profile, recurrent):
    writes = [f'/记 {owner} | {field} | {value}' for owner, field, value in RECORDS]
    written, write_seconds = replay(profile, writes, recurrent)
    if any((row['kind'], row.get('record')) != ('saved', number) for number, row in enumerate(written, 1)):
        raise ValueError(f'One or more test records could not be written: {written}')
    database = profile / 'versions.sqlite3'
    before = digest(database)
    requests = [(mode, prompt, expected if mode != 'none' else None)
                for expected, *prompts in PAIRS
                for mode, prompt in zip(('explicit', 'natural', 'none'), prompts, strict=True)]
    questions = [item for _, prompt, _ in requests for item in ('/清空对话', prompt)]
    answers, seconds = replay(profile, questions, recurrent)
    if any(answers[index]['kind'] != 'reset' for index in range(0, len(questions), 2)):
        raise ValueError('The replay did not clear context between prompts')
    rows = []
    values = [record[-1] for record in RECORDS]
    for index, (mode, prompt, expected) in enumerate(requests):
        response = answers[index * 2 + 1]
        selected = response.get('selected', [])
        shown = [value for value in values if value in response['answer']]
        correct_selection = selected == ([expected] if expected else [])
        correct_values = shown == ([values[expected - 1]] if expected else [])
        passed = correct_selection and correct_values and (expected is None or response['kind'] == 'read')
        rows.append(dict(mode=mode, prompt=prompt, expected=expected, response=response, shown=shown, passed=passed))
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        intact = (connection.execute('pragma integrity_check').fetchone() == ('ok',)
                  and connection.execute('select count(*) from versions').fetchone() == (len(RECORDS),))
    if not intact or digest(database) != before:
        raise ValueError('The read-only replay changed the version archive')
    summary = {mode: sum(row['passed'] for row in rows if row['mode'] == mode)
               for mode in ('explicit', 'natural', 'none')}
    summary['natural_asked'] = sum(row['mode'] == 'natural' and row['response']['kind'] == 'clarify' for row in rows)
    print('recurrent' if recurrent else 'default', summary, flush=True)
    return dict(rows=rows, summary=summary, archive_sha256=before, write_seconds=write_seconds, seconds=seconds)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    results = {'default': evaluate((args.output / 'default').resolve(), None),
               'recurrent': evaluate((args.output / 'recurrent').resolve(), args.recurrent)}
    store_json(args.output / 'result.json', {**results, 'script_sha256': digest(Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--recurrent', required=True, nargs=3, type=Path)
    main(parser.parse_args())
