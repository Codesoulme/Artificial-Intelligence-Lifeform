"""Kill real copied-core variable-length transactions before/after SQLite commit."""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_chunk_variable_length_pilot_20261117 import (
    DATABASE as SOURCE, INPUT, NAMES, resources, target_versions)
from scripts.deepseek_spora_chunked_memory_pilot_20261117 import first_user_turn
from scripts.longmemeval_no_experience_diagnostic_20261117 import select_cases
from spora.research_chunk_history_store import ResearchChunkHistoryStore

DIR = ROOT / 'runs/deepseek-spora-variable-crash-20261117'
OUT = DIR / 'result.json'
STAGES = {'before': 'after_first_tensor', 'after': 'after_commit'}


def desired(tokenizer):
    source = tokenizer.encode(first_user_turn(select_cases(INPUT)[2]), add_special_tokens=False)
    if len(source) < 36:
        raise ValueError('Target is too short')
    return source[:36]


def backup(source, target):
    with target.open('xb'):
        pass
    src = sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def worker(which):
    tokenizer, codec = resources()
    store = ResearchChunkHistoryStore.open(DIR / f'{which}.sqlite', codec)
    identity = tokenizer.encode(NAMES[0], add_special_tokens=False)
    def kill(stage):
        if stage == STAGES[which]:
            os._exit(77)
    try:
        store.write(0, [0, 1, 2, 3, 4, 5, 30, 31, 32], identity, desired(tokenizer),
                    allow_update=True, hook=kill)
    finally:
        store.close()
    raise RuntimeError('Fault hook was not reached')


def verify(which, tokenizer, codec):
    store = ResearchChunkHistoryStore.open(DIR / f'{which}.sqlite', codec)
    try:
        old, short, long = target_versions(tokenizer)
        expected = [old[0] + [short, long] + ([desired(tokenizer)] if which == 'after' else [])] + old[1:]
        identities = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
        rows = []
        for person, versions in enumerate(expected):
            current = store.read(person)
            rows.append({'record': person, 'mode': 'current',
                         'exact': current == {'identity_ids': identities[person], 'fact_ids': versions[-1]}})
            for version, tokens in enumerate(versions):
                result = store.read(person, version)
                rows.append({'record': person, 'mode': 'history', 'version': version,
                             'exact': result == {'identity_ids': identities[person], 'fact_ids': tokens}})
        record = next(r for r in store.metadata['records'] if r['id'] == 0)
        expected_cursor = 53 if which == 'before' else 62
        expected_keys = [0, 1, 2, 3, 4, 5, 30, 31] + ([32] if which == 'after' else [])
        return {'cursor': store.state.cursor, 'cursor_ok': store.state.cursor == expected_cursor,
                'reserved_keys_ok': record['keys'] == expected_keys, 'rows': rows,
                'current_exact': sum(r['exact'] for r in rows if r['mode'] == 'current'),
                'current_total': sum(r['mode'] == 'current' for r in rows),
                'history_exact': sum(r['exact'] for r in rows if r['mode'] == 'history'),
                'history_total': sum(r['mode'] == 'history' for r in rows)}
    finally:
        store.close()


def main():
    if DIR.exists():
        raise FileExistsError(DIR)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_sha, input_sha = runtime.sha(SOURCE), runtime.sha(INPUT)
    tokenizer, codec = resources()
    DIR.mkdir(parents=True)
    results = {}
    started = time.perf_counter()
    for which in ('before', 'after'):
        backup(SOURCE, DIR / f'{which}.sqlite')
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', which],
                               cwd=ROOT, capture_output=True, text=True, timeout=180)
        if child.returncode != 77:
            raise RuntimeError(f'{which} worker exit {child.returncode}: {child.stderr[-2000:]}')
        results[which] = verify(which, tokenizer, codec)
        print(json.dumps({'stage': which, 'exit_code': child.returncode,
                          'cursor_ok': results[which]['cursor_ok'],
                          'current': results[which]['current_exact'],
                          'history': results[which]['history_exact']}), flush=True)
    if runtime.sha(SOURCE) != source_sha or runtime.sha(INPUT) != input_sha:
        raise ValueError('Protected source changed')
    report = {'status': 'isolated_copied_core_process_kill', 'results': results,
              'source_sha256': source_sha, 'input_sha256': input_sha,
              'seconds_after_initial_codec': time.perf_counter()-started,
              'limitations': ['Copied uncontrolled temporal core, not formal individual.',
                              'One simulated 32-to-36-token update on each isolated copy.',
                              'Abrupt process exits at application hooks, not power loss or hardware faults.',
                              'Explicit identity, numeric addresses, versions and event times.',
                              'No natural language routing, provenance guard, or QA.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'completed': True}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=tuple(STAGES))
    args = parser.parse_args()
    worker(args.worker) if args.worker else main()
