"""Verify multi-turn recall of newly written neural memories across two processes."""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from deepseek_awake_recall_intent import DEFAULT_DATABASE
from deepseek_continuous_memory_chat import load_chat
from deepseek_output_aligned_memory import ROOT, sha


WRITES = {'drink': '米香玄米茶', 'work': '标本整理员', 'place': '雪松阅览室'}


def write_copy(database):
    chat = load_chat(database)
    for field, value in WRITES.items():
        result = chat.remember(26, field, value)
        if result['action'] != 'stored':
            raise ValueError(f'Expected a new write: {result}')


def expect(row, **expected):
    if any(row.get(key) != value for key, value in expected.items()):
        raise AssertionError(f'Expected {expected}, got {row}')


def verify(source, recall_head, subject_head):
    original_digest = sha(source)
    env = dict(os.environ, PYTHONUTF8='1')
    with TemporaryDirectory(prefix='spora-awake-reload-') as directory:
        temp = Path(directory)
        copy = temp / 'memory.sqlite3'
        with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as original:
            with closing(sqlite3.connect(copy)) as destination:
                original.backup(destination)
        subprocess.run([sys.executable, __file__, '--write-copy', str(copy)], cwd=ROOT, env=env, check=True, timeout=150)
        written_digest = sha(copy)
        args = [sys.executable, '-u', str(ROOT / 'scripts/deepseek_continuous_memory_chat.py'),
                '--memory-file', str(copy), '--awake-state', str(temp / 'awake.pt'), '--awake-topic-recall',
                '--awake-recall-intent-head', str(recall_head.resolve()), '--awake-subject-intent-head',
                str(subject_head.resolve()), '--awake-subject-hops', '3', '--inherited-router', '--retrained-gate',
                '--earlier-readout', '--json']
        questions = ('齐宁平常最爱喝什么？', '那他的职业是什么？', '那他去过哪里？', '她最喜欢喝什么？',
                     '上一次记录里，齐宁平常最爱喝什么？', '那他的职业是什么？', '温谨平常最爱喝什么？')
        run = subprocess.run(args, cwd=ROOT, env=env, input='\n'.join(questions) + '\n', text=True, encoding='utf-8',
                             stdout=subprocess.PIPE, check=True, timeout=150)
        named, work, place, drink, archived, after_archive, neighbor = map(json.loads, run.stdout.splitlines())
        expect(named, action='read', entity=26, field='drink', confirmed_recent_read=True)
        for row, field in ((work, 'work'), (place, 'place'), (drink, 'drink')):
            expect(row, action='read', entity=26, field=field, response=WRITES[field], subject_reason='awake_same_subject')
        expect(archived, action='read', entity=26, field='drink', corrections_back=1, confirmed_recent_read=False)
        expect(after_archive, action='clarify')
        expect(neighbor, action='read', entity=65, field='drink')
        for row, value in ((named, WRITES['drink']), (archived, '青杏冰茶'), (neighbor, '桂花乌龙')):
            if value not in row['response']:
                raise AssertionError(f'Expected {value!r} in {row}')
        if sha(copy) != written_digest or sha(source) != original_digest:
            raise ValueError('A memory file changed during the read-only verification')
    print('Verified cross-process writes, implicit field switches, a correction, and an adjacent identity.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--recall-head', type=Path)
    parser.add_argument('--subject-head', type=Path)
    parser.add_argument('--write-copy', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.write_copy is not None:
        write_copy(args.write_copy)
        return
    if args.recall_head is None or args.subject_head is None:
        parser.error('--recall-head and --subject-head are required')
    verify(args.source, args.recall_head, args.subject_head)


if __name__ == '__main__':
    main()
