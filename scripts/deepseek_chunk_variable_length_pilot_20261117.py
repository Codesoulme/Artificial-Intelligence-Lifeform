"""Copy a research snapshot and test shrink/growth; never modify its source."""
import argparse
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from transformers import AutoTokenizer

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_chunk_history_durable_pilot_20261117 import (
    DATABASE as SOURCE_DB, INPUT, NAMES, codec_from_embedding, expected_rows)
from scripts.deepseek_spora_chunked_memory_pilot_20261117 import first_user_turn
from scripts.deepseek_spora_prefix_features import read_bf16_tensor
from scripts.longmemeval_no_experience_diagnostic_20261117 import select_cases
from spora.language_training import frozen_digest
from spora.research_chunk_history_store import ResearchChunkHistoryStore

DIR = ROOT / 'runs/deepseek-spora-chunk-variable-length-20261117'
DATABASE = DIR / 'research.sqlite'
WRITE_REPORT = DIR / 'write.json'
READ_REPORT = DIR / 'read.json'


def resources():
    weights = runtime.MODEL / 'model.safetensors'
    if runtime.sha(weights) != runtime.WEIGHT_SHA:
        raise ValueError('Base mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    with weights.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if header_size > 100_000_000 or header_size > weights.stat().st_size - 8:
            raise ValueError('Invalid header')
        header = json.loads(stream.read(header_size))
        embedding = read_bf16_tensor(stream, header, 8 + header_size, weights.stat().st_size,
                                     'model.embed_tokens.weight').to(torch.float16)
    codec = codec_from_embedding(embedding, tokenizer)
    del embedding
    return tokenizer, codec


def target_versions(tokenizer):
    old = expected_rows(tokenizer)
    source = tokenizer.encode(first_user_turn(select_cases(INPUT)[2]), add_special_tokens=False)
    if len(source) < 56:
        raise ValueError('Target is too short')
    return old, source[:12], source[24:56]


def write_phase():
    if DIR.exists():
        raise FileExistsError(DIR)
    if not SOURCE_DB.is_file():
        raise FileNotFoundError(SOURCE_DB)
    before = runtime.sha(SOURCE_DB)
    input_sha = runtime.sha(INPUT)
    tokenizer, codec = resources()
    old, short, long = target_versions(tokenizer)
    identity = tokenizer.encode(NAMES[0], add_special_tokens=False)
    DIR.mkdir(parents=True)
    # SQLite backup captures a consistent source snapshot, including any WAL.
    # Exclusive creation prevents accidental overwrite of another experiment.
    with DATABASE.open('xb'):
        pass
    source = sqlite3.connect(SOURCE_DB.resolve().as_uri() + '?mode=ro', uri=True)
    destination = sqlite3.connect(DATABASE)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    started = time.perf_counter()
    store = ResearchChunkHistoryStore.open(DATABASE, codec)
    try:
        if store.metadata['format'] != 'spora-research-chunk-history-v1' or store.state.cursor != 42:
            raise ValueError('Unexpected source state')
        if store.read(0)['fact_ids'] != old[0][-1]:
            raise ValueError('Source current does not match expected')
        shrink = store.write(0, [0, 1, 2], identity, short, allow_update=True)
        expand = store.write(0, [0, 1, 2, 3, 4, 5, 30, 31], identity, long, allow_update=True)
        digest, cursor, revision = frozen_digest(store.model, store.state), store.state.cursor, store.store.revision
        reserved = next(r for r in store.metadata['records'] if r['id'] == 0)['keys']
    finally:
        store.close()
    if runtime.sha(SOURCE_DB) != before or runtime.sha(INPUT) != input_sha:
        raise ValueError('Source changed')
    report = {'status': 'isolated_copy_updated', 'source_sha256': before, 'input_sha256': input_sha,
              'base_sha256': runtime.WEIGHT_SHA, 'state_digest': digest, 'cursor': cursor,
              'revision': revision, 'reserved_keys': reserved, 'shrink': shrink, 'expand': expand,
              'seconds_after_backup': time.perf_counter()-started}
    with WRITE_REPORT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'cursor': cursor, 'revision': revision, 'reserved_keys': reserved}), flush=True)


def read_phase():
    if READ_REPORT.exists():
        raise FileExistsError(READ_REPORT)
    report = json.loads(WRITE_REPORT.read_text(encoding='utf-8'))
    if runtime.sha(INPUT) != report['input_sha256'] or runtime.sha(SOURCE_DB) != report['source_sha256']:
        raise ValueError('Source changed')
    tokenizer, codec = resources()
    old, short, long = target_versions(tokenizer)
    expected = [old[0] + [short, long]] + old[1:]
    names = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
    store = ResearchChunkHistoryStore.open(DATABASE, codec)
    rows = []
    try:
        if (frozen_digest(store.model, store.state) != report['state_digest']
                or store.state.cursor != report['cursor'] or store.store.revision != report['revision']):
            raise ValueError('Snapshot differs from write process')
        for person, versions in enumerate(expected):
            current = store.read(person)
            rows.append({'record': person, 'mode': 'current',
                         'exact': current == {'identity_ids': names[person], 'fact_ids': versions[-1]}})
            for version, tokens in enumerate(versions):
                answer = store.read(person, version)
                rows.append({'record': person, 'mode': 'history', 'version': version,
                             'exact': answer == {'identity_ids': names[person], 'fact_ids': tokens}})
    finally:
        store.close()
    result = {'status': 'separate_process_variable_length_recovery', 'rows': rows,
              'limitations': ['Fixed explicit address list and event versions, not natural language selection.',
                              'Copied uncontrolled research state; no formal individual modified.',
                              'Only one shrink/grow sequence and five previously inspected examples.',
                              'Short and long versions are simulated, not factual corrections.']}
    with READ_REPORT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'current': sum(r['exact'] for r in rows if r['mode'] == 'current'),
                      'current_total': sum(r['mode'] == 'current' for r in rows),
                      'history': sum(r['exact'] for r in rows if r['mode'] == 'history'),
                      'history_total': sum(r['mode'] == 'history' for r in rows)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('write', 'read'))
    args = parser.parse_args()
    write_phase() if args.phase == 'write' else read_phase()
