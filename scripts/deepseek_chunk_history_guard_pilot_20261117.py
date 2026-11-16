"""Replay a known corrupting write against the new pre-commit stability guard."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_chunk_variable_crash_pilot_20261117 import (
    SOURCE, INPUT, NAMES, backup, desired, resources, target_versions)
from spora.language_training import frozen_digest
from spora.research_chunk_history_store import MemoryIntegrityError, ResearchChunkHistoryStore

DIR = ROOT / 'runs/deepseek-spora-chunk-history-guard-20261117'


def main():
    if DIR.exists():
        raise FileExistsError(DIR)
    source_sha, input_sha = runtime.sha(SOURCE), runtime.sha(INPUT)
    tokenizer, codec = resources()
    old, short, long = target_versions(tokenizer)
    expected = [old[0] + [short, long]] + old[1:]
    identities = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
    DIR.mkdir(parents=True)
    database = DIR / 'guarded.sqlite'
    backup(SOURCE, database)
    start = time.perf_counter()
    store = ResearchChunkHistoryStore.open(database, codec)
    commit_hook_called = []
    try:
        original = frozen_digest(store.model, store.state)
        original_cursor, original_revision = store.state.cursor, store.store.revision
        try:
            store.write(0, [0, 1, 2, 3, 4, 5, 30, 31, 32], identities[0], desired(tokenizer),
                        allow_update=True, hook=lambda stage: commit_hook_called.append(stage))
        except MemoryIntegrityError as exc:
            reason = str(exc)
        else:
            raise RuntimeError('Known corrupting write was not rejected')
        if (frozen_digest(store.model, store.state) != original or store.state.cursor != original_cursor
                or store.store.revision != original_revision or commit_hook_called):
            raise ValueError('Guard changed or attempted to commit the original snapshot')
    finally:
        store.close()
    reloaded = ResearchChunkHistoryStore.open(database, codec)
    rows = []
    try:
        if (frozen_digest(reloaded.model, reloaded.state) != original
                or reloaded.state.cursor != original_cursor or reloaded.store.revision != original_revision):
            raise ValueError('Durable snapshot changed')
        for person, versions in enumerate(expected):
            current = reloaded.read(person)
            rows.append({'record': person, 'mode': 'current',
                         'exact': current == {'identity_ids': identities[person], 'fact_ids': versions[-1]}})
            for version, tokens in enumerate(versions):
                value = reloaded.read(person, version)
                rows.append({'record': person, 'mode': 'history', 'version': version,
                             'exact': value == {'identity_ids': identities[person], 'fact_ids': tokens}})
    finally:
        reloaded.close()
    if runtime.sha(SOURCE) != source_sha or runtime.sha(INPUT) != input_sha:
        raise ValueError('Source changed')
    result = {'status': 'known_corrupting_write_rejected', 'reason': reason, 'rows': rows,
              'commit_hook_called': bool(commit_hook_called), 'cursor': original_cursor,
              'revision': original_revision, 'source_sha256': source_sha, 'input_sha256': input_sha,
              'seconds_after_backup_and_codec': time.perf_counter()-start,
              'limitations': ['One previously observed failure replayed on copied uncontrolled state.',
                              'Guard compares existing decoded content, not external truth.',
                              'All retained records are reread; scaling and real-time cost not established.',
                              'No formal identity/source policy, automatic question routing or QA.']}
    with (DIR / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'reason': reason, 'commit_called': bool(commit_hook_called),
                      'current': sum(r['exact'] for r in rows if r['mode'] == 'current'),
                      'current_total': sum(r['mode'] == 'current' for r in rows),
                      'history': sum(r['exact'] for r in rows if r['mode'] == 'history'),
                      'history_total': sum(r['mode'] == 'history' for r in rows)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
