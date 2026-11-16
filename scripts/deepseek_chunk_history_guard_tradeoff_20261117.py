"""Probe accepted and rejected sizes on independent copies of the same state."""
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

DIR = ROOT / 'runs/deepseek-spora-chunk-history-guard-tradeoff-20261117'


def main():
    if DIR.exists():
        raise FileExistsError(DIR)
    source_sha, input_sha = runtime.sha(SOURCE), runtime.sha(INPUT)
    tokenizer, codec = resources()
    initial, short, long = target_versions(tokenizer)
    baseline = [initial[0] + [short, long]] + initial[1:]
    identities = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
    tokens = desired(tokenizer)
    DIR.mkdir(parents=True)
    rows = []
    for length in (4, 12, 24, 36):
        path = DIR / f'case-{length}.sqlite'
        backup(SOURCE, path)
        store = ResearchChunkHistoryStore.open(path, codec)
        start = time.perf_counter()
        before = frozen_digest(store.model, store.state)
        before_cursor, before_revision = store.state.cursor, store.store.revision
        keys = ([0, 1, 2, 3, 4, 5, 30, 31, 32])[:(length+3)//4]
        try:
            try:
                store.write(0, keys, identities[0], tokens[:length], allow_update=True)
                accepted, reason = True, None
            except MemoryIntegrityError as error:
                accepted, reason = False, str(error)
            elapsed = time.perf_counter()-start
            after = frozen_digest(store.model, store.state)
            if not accepted and (before != after or store.state.cursor != before_cursor
                                 or store.store.revision != before_revision):
                raise ValueError('Rejected write changed in-memory snapshot')
        finally:
            store.close()
        reopened = ResearchChunkHistoryStore.open(path, codec)
        try:
            if frozen_digest(reopened.model, reopened.state) != after:
                raise ValueError('Durable state differs from in-memory snapshot')
            histories = []
            currents = []
            expected = [baseline[0] + ([tokens[:length]] if accepted else [])] + baseline[1:]
            for person, versions in enumerate(expected):
                currents.append(reopened.read(person) == {'identity_ids': identities[person], 'fact_ids': versions[-1]})
                for index, value in enumerate(versions):
                    histories.append(reopened.read(person, index) == {'identity_ids': identities[person], 'fact_ids': value})
            row = {'length': length, 'accepted': accepted, 'reason': reason, 'write_seconds': elapsed,
                   'current_exact': sum(currents), 'current_total': len(currents),
                   'history_exact': sum(histories), 'history_total': len(histories),
                   'reloaded_cursor': reopened.state.cursor, 'reloaded_revision': reopened.store.revision}
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            reopened.close()
    if runtime.sha(SOURCE) != source_sha or runtime.sha(INPUT) != input_sha:
        raise ValueError('Protected source changed')
    report = {'status': 'guard_tradeoff_development_probe', 'rows': rows,
              'source_sha256': source_sha, 'input_sha256': input_sha,
              'limitations': ['Four lengths from one previously inspected input and one source state.',
                              'The 36-token failure was discovered before this guard was built.',
                              'Identity, addresses, and update intent supplied manually.',
                              'Times exclude codec reconstruction, database copy, and full verification.',
                              'Passing checks compare decoded retained contents, not external truth.']}
    with (DIR / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'completed': len(rows)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
