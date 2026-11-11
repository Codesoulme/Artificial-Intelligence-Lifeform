"""Compare optional tensor event-key index with the neural-only scan."""
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts import spora_event_scan_selector as selector
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_cross_key_history_probe import OUTPUT as PRIOR
from scripts.deepseek_spora_delayed_fact_audit import SCENARIOS, load_individual_from_verified_checkpoint
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, save
from spora.historical_resolutions import HistoricalResolutions
from spora.internal_memory import enable_event_key_index, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-indexed-history-audit-20261112.json'
LEVELS = (0, 64, 128, 256, 512, 768)


def same_neural_state(a, b):
    if a.cursor != b.cursor or len(a.pages) != len(b.pages) or a.internal.metadata() != b.internal.metadata():
        return False
    pairs = list(zip(a.current, b.current)) + list(zip(a.internal.provenance, b.internal.provenance))
    pairs.append((a.internal.evidence, b.internal.evidence))
    for page_a, page_b in zip(a.pages, b.pages):
        for site_a, site_b in zip(page_a, page_b):
            pairs.extend(zip(site_a, site_b))
    return all(torch.equal(x, y) for x, y in pairs)


def timed_select(model, state, key):
    durations, outputs, scanned = [], [], []
    for _ in range(3):
        with patch.object(selector, 'read_provenance_batch', wraps=selector.read_provenance_batch) as read:
            started = time.perf_counter()
            result = selector.select_event(model, state, key, 'effective_previous', as_of=state.cursor - 1)
            durations.append(time.perf_counter() - started)
        outputs.append(result)
        scanned.append(len(read.call_args.args[2]))
    if not all(value == outputs[0] for value in outputs) or len(set(scanned)) != 1:
        raise ValueError('Repeated selection changed')
    return {'result': outputs[0], 'median_seconds': statistics.median(durations), 'samples_seconds': durations,
            'events_scanned': scanned[0],
            'correct': outputs[0]['status'] == 'experimental_candidate' and outputs[0].get('event_time') == 4}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if (prior['source_checkpoint_sha256'] != checkpoint_sha or prior['summary']['first_problem_is_target']
            or prior['summary']['first_problem_event'] != 94):
        raise ValueError('Prior cross-key evidence mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    model.historical_resolutions = HistoricalResolutions(model.config.key_count, 4)
    regular = fresh_controlled_state(model)
    indexed = enable_event_key_index(model, fresh_controlled_state(model))
    key, distractor = ENROLLMENTS[0][0], ENROLLMENTS[3][0]

    def write_pair(subject_key, value, valid_at=None):
        nonlocal regular, indexed
        if regular.cursor == capacity(model, regular):
            regular, indexed = grow(model, regular), grow(model, indexed)
        regular, a = observe_sourced(model, regular, subject_key, value, 'direct', 1.0, valid_at=valid_at)
        indexed, b = observe_sourced(model, indexed, subject_key, value, 'direct', 1.0, valid_at=valid_at)
        if a != b:
            raise ValueError('Write decisions diverged')

    for subject, value, valid_at in SCENARIOS[1]:
        write_pair(key if subject == 'T' else distractor, value, valid_at)
    model.historical_resolutions.record(model, indexed, key, 4, value=3, valid_at=1, decision='confirm')
    noise_keys = [k for k in range(10, 51) if k not in (key, distractor)]
    count, rows = 0, []
    for level in LEVELS:
        while count < level:
            write_pair(noise_keys[count % len(noise_keys)], (count * 7 + 2) % model.config.unknown)
            count += 1
        if not same_neural_state(regular, indexed):
            raise ValueError('Index changed neural state')
        before = timed_select(model, regular, key)
        after = timed_select(model, indexed, key)
        row = {'noise_writes': level, 'cursor': indexed.cursor, 'pages': len(indexed.pages),
               'index_bytes': sum(x.numel() * x.element_size() for x in indexed.internal.event_keys),
               'neural_state_exact_equal': True, 'regular': before, 'indexed': after}
        rows.append(row)
        print(json.dumps({'noise': level, 'cursor': indexed.cursor, 'index_bytes': row['index_bytes'],
                          'regular_status': before['result']['status'], 'regular_correct': before['correct'],
                          'regular_scanned': before['events_scanned'], 'regular_seconds': before['median_seconds'],
                          'indexed_status': after['result']['status'], 'indexed_correct': after['correct'],
                          'indexed_scanned': after['events_scanned'], 'indexed_seconds': after['median_seconds']}), flush=True)
    state_digest = frozen_digest(model, indexed)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'individual.pt'
        save(path, model, indexed, {'purpose': 'experimental-event-key-index-audit'})
        restored_model, restored_state, _ = load(path, 'cpu')
        after_reload = timed_select(restored_model, restored_state, key)
        reload_same = (frozen_digest(restored_model, restored_state) == state_digest and after_reload['correct']
                       and after_reload['result'] == rows[-1]['indexed']['result'])
    del model.historical_resolutions
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    result = {'status': 'controlled_event_key_index_experiment', 'rows': rows, 'same_process_reload_same': reload_same,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['One synthetic trajectory and one confirmed subject, not a general identity benchmark.',
                              'Exact per-event key metadata is an auxiliary tensor index, not learned neural weights.',
                              'Only fresh controlled states can opt in; old histories cannot be safely backfilled.',
                              'Index validation checks structure, not the truth of a tampered but valid key entry.',
                              'Neural value and provenance recall can still fail or return uncertain results.',
                              'CPU timings exclude language generation, I/O and persistent write overhead.',
                              'Three repeated measurements per point; no isolated benchmarking.',
                              'Same-process reload, no crash, no natural-language writing or real user state.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'levels': len(rows), 'indexed_all_correct': all(r['indexed']['correct'] for r in rows),
                      'same_process_reload_same': reload_same}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
