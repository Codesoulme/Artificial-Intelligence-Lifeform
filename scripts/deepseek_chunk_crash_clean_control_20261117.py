"""Compare a clean update with the same committed update after process kill."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_chunk_variable_crash_pilot_20261117 import (
    DIR as CRASH_DIR, INPUT, SOURCE, NAMES, backup, desired, resources, target_versions)
from spora.language_training import frozen_digest
from spora.research_chunk_history_store import ResearchChunkHistoryStore

DIR = ROOT / 'runs/deepseek-spora-chunk-clean-control-20261117'


def target_metrics(store, tokenizer, versions):
    identity = tokenizer.encode(NAMES[0], add_special_tokens=False)
    rows = []
    for version, tokens in enumerate(versions):
        decoded = store.read(0, version)
        actual = decoded['fact_ids']
        rows.append({'version': version, 'identity_exact': decoded['identity_ids'] == identity,
                     'expected_length': len(tokens), 'actual_length': len(actual),
                     'mismatch_positions': [i for i, (x, y) in enumerate(zip(actual, tokens)) if x != y],
                     'exact': actual == tokens and decoded['identity_ids'] == identity})
    return rows


def main():
    if DIR.exists():
        raise FileExistsError(DIR)
    before = runtime.sha(SOURCE)
    tokenizer, codec = resources()
    old, short, long = target_versions(tokenizer)
    versions = old[0] + [short, long, desired(tokenizer)]
    DIR.mkdir(parents=True)
    database = DIR / 'clean.sqlite'
    backup(SOURCE, database)
    started = time.perf_counter()
    clean = ResearchChunkHistoryStore.open(database, codec)
    killed = None
    try:
        identity = tokenizer.encode(NAMES[0], add_special_tokens=False)
        stats = clean.write(0, [0, 1, 2, 3, 4, 5, 30, 31, 32], identity,
                            versions[-1], allow_update=True)
        clean_rows = target_metrics(clean, tokenizer, versions)
        killed = ResearchChunkHistoryStore.open(CRASH_DIR / 'after.sqlite', codec)
        killed_rows = target_metrics(killed, tokenizer, versions)
        state_equal = frozen_digest(clean.model, clean.state) == frozen_digest(killed.model, killed.state)
        metadata_equal = clean.metadata == killed.metadata
        revision_equal = clean.store.revision == killed.store.revision
    finally:
        if killed is not None:
            killed.close()
        clean.close()
    if runtime.sha(SOURCE) != before:
        raise ValueError('Source changed')
    result = {'status': 'clean_vs_killed_control', 'clean_rows': clean_rows, 'killed_rows': killed_rows,
              'state_digest_equal': state_equal, 'metadata_equal': metadata_equal,
              'revision_equal': revision_equal, 'write_stats': stats['stats'],
              'seconds_after_codec': time.perf_counter()-started, 'source_sha256': before,
              'limitations': ['Single deterministic ordering on copied uncontrolled model core.',
                              'An equal state does not prove all kinds of crash are safe.',
                              'Only controlled versions and explicitly specified addresses.']}
    with (DIR / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'state_equal': state_equal, 'metadata_equal': metadata_equal,
                      'revision_equal': revision_equal,
                      'clean_inexact': [r['version'] for r in clean_rows if not r['exact']],
                      'killed_inexact': [r['version'] for r in killed_rows if not r['exact']]}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
