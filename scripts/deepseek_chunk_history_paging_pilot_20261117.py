"""Create isolated research state; attempt 65 events and reload separately."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_chunk_history_ridge_holdout_20261117 import fixed_inputs
from scripts.deepseek_chunk_variable_crash_pilot_20261117 import SOURCE, INPUT, NAMES, resources
from spora.language_training import frozen_digest
from spora.research_chunk_history_store import MemoryIntegrityError, ResearchChunkHistoryStore

DIR = ROOT / 'runs/deepseek-spora-chunk-history-paging-20261117'
DATABASE = DIR / 'research.sqlite'
WRITE_REPORT = DIR / 'write.json'
READ_REPORT = DIR / 'read.json'


def write_phase():
    if DIR.exists():
        raise FileExistsError(DIR)
    paths = {'source': SOURCE, 'input': INPUT, 'base': runtime.MODEL / 'model.safetensors'}
    hashes = {name: runtime.sha(path) for name, path in paths.items()}
    tokenizer, codec = resources()
    inputs = fixed_inputs(INPUT, tokenizer)
    names = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
    reference = ResearchChunkHistoryStore.open(SOURCE, codec)
    source_digest = frozen_digest(reference.model, reference.state)
    DIR.mkdir(parents=True)
    experimental = None
    refs = [[] for _ in NAMES]
    steps = []
    start = time.perf_counter()
    try:
        experimental = ResearchChunkHistoryStore.create(DATABASE, reference.model, codec)
        # Five new records, five updates, then two updates of the first record.
        schedule = [(person, person, 24) for person in range(5)]
        schedule += [(person, 5 + person, 24) for person in range(5)]
        schedule += [(0, 10, 8), (0, 10, 12)]
        for person, input_index, length in schedule:
            keys = list(range(person*6, person*6 + (length+3)//4))
            begin = time.perf_counter()
            try:
                result = experimental.write(person, keys, names[person], inputs[input_index]['tokens'][:length],
                                            allow_update=bool(refs[person]))
            except MemoryIntegrityError as exc:
                steps.append({'person': person, 'input_index': input_index, 'length': length,
                              'accepted': False, 'reason': str(exc),
                              'cursor': experimental.state.cursor, 'pages': len(experimental.state.pages),
                              'seconds': time.perf_counter()-begin})
                print(json.dumps(steps[-1]), flush=True)
                break
            refs[person].append({'input_index': input_index, 'length': length})
            steps.append({'person': person, 'input_index': input_index, 'length': length,
                          'accepted': True, 'cursor': experimental.state.cursor,
                          'pages': len(experimental.state.pages), 'times': result['times'],
                          'tensor_bytes_written': result['stats']['tensor_bytes_written'],
                          'seconds': time.perf_counter()-begin})
            print(json.dumps(steps[-1]), flush=True)
        state_digest = frozen_digest(experimental.model, experimental.state)
        cursor, revision, pages = experimental.state.cursor, experimental.store.revision, len(experimental.state.pages)
        if frozen_digest(reference.model, reference.state) != source_digest:
            raise ValueError('Source individual changed')
    finally:
        if experimental is not None:
            experimental.close()
        reference.close()
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Source file changed')
    report = {'status': 'isolated_paging_pilot_write', 'refs': refs, 'steps': steps,
              'state_digest': state_digest, 'cursor': cursor, 'revision': revision, 'pages': pages,
              'source_sha256': hashes, 'seconds_after_codec': time.perf_counter()-start,
              'selected_case_ids': [item['case_id'] for item in inputs]}
    with WRITE_REPORT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'cursor': cursor, 'pages': pages, 'accepted': sum(s['accepted'] for s in steps)}), flush=True)


def read_phase():
    if READ_REPORT.exists():
        raise FileExistsError(READ_REPORT)
    report = json.loads(WRITE_REPORT.read_text(encoding='utf-8'))
    if runtime.sha(INPUT) != report['source_sha256']['input'] or runtime.sha(SOURCE) != report['source_sha256']['source']:
        raise ValueError('Source changed')
    tokenizer, codec = resources()
    inputs = fixed_inputs(INPUT, tokenizer)
    if [item['case_id'] for item in inputs] != report['selected_case_ids']:
        raise ValueError('Input selection changed')
    names = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
    store = ResearchChunkHistoryStore.open(DATABASE, codec)
    rows = []
    try:
        if (frozen_digest(store.model, store.state) != report['state_digest'] or
                (store.state.cursor, store.store.revision, len(store.state.pages)) !=
                (report['cursor'], report['revision'], report['pages'])):
            raise ValueError('Restored state mismatch')
        for person, versions in enumerate(report['refs']):
            if not versions:
                continue
            for version, ref in enumerate(versions):
                expected = {'identity_ids': names[person],
                            'fact_ids': inputs[ref['input_index']]['tokens'][:ref['length']]}
                rows.append({'record': person, 'mode': 'history', 'version': version,
                             'exact': store.read(person, version) == expected})
            last = versions[-1]
            expected = {'identity_ids': names[person],
                        'fact_ids': inputs[last['input_index']]['tokens'][:last['length']]}
            rows.append({'record': person, 'mode': 'current', 'exact': store.read(person) == expected})
    finally:
        store.close()
    result = {'status': 'isolated_separate_process_paging_reload', 'rows': rows,
              'cursor': report['cursor'], 'pages': report['pages'],
              'limitations': ['Copied uncontrolled model, not formal individual.',
                              'Explicit identity, address, version and token boundaries.',
                              'One known sequence; stops at first rejected write.',
                              'No natural language selection, model generation or provenance checks.']}
    with READ_REPORT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'cursor': report['cursor'], 'pages': report['pages'],
                      'current_exact': sum(r['exact'] for r in rows if r['mode'] == 'current'),
                      'current_total': sum(r['mode'] == 'current' for r in rows),
                      'history_exact': sum(r['exact'] for r in rows if r['mode'] == 'history'),
                      'history_total': sum(r['mode'] == 'history' for r in rows)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('write', 'read'))
    args = parser.parse_args()
    write_phase() if args.phase == 'write' else read_phase()
