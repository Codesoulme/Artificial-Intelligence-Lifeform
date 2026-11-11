"""Experimental provenance uncertainty band on fresh dense trajectories."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_delayed_fact_audit import historical, load_individual_from_verified_checkpoint
from scripts.deepseek_spora_provenance_interleaved_probe import OUTPUT as PRIOR
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import ACTIONS, SOURCES, enable_event_key_index, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-provenance-band-selector-pilot-20261112.json'
KEYS = (0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14)


def experimental_decode(result, event_time, encoding):
    """Local-only 0.3/0.7 band; preserve all other production checks."""
    if encoding != 2:
        raise ValueError('Pilot only supports bit provenance')
    source, action = result[:4], result[4:9]
    if float(source.max()) < 0.7 or float(action.max()) < 0.7:
        return {'source': None, 'reason': 'uncertain_provenance'}
    labels = {'source': SOURCES[int(source.argmax())], 'action': ACTIONS[int(action.argmax())]}
    bits = result[9:] / source.sum().clamp_min(0.1)
    if bool(((bits > 0.3) & (bits < 0.7)).any()):
        return dict(labels, quality=None, valid_at=None, reason='uncertain_numeric_provenance')
    values = (bits > 0.5).tolist()
    quality = sum(int(bit) << i for i, bit in enumerate(values[:8]))
    valid_at = sum(int(bit) << i for i, bit in enumerate(values[8:]))
    if valid_at > event_time:
        return dict(labels, quality=None, valid_at=None, reason='invalid_numeric_provenance')
    return dict(labels, quality=quality / 255, valid_at=valid_at, quality_resolution=1 / 255)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    if json.loads(PRIOR.read_text(encoding='utf-8'))['source_checkpoint_sha256'] != source_sha:
        raise ValueError('Prior checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    rows = []
    for key in KEYS:
        state = enable_event_key_index(model, fresh_controlled_state(model))
        writes = []
        for event in range(64):
            value = (1, 6, 3, 4)[(event + key) % 4]
            state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
            writes.append({'event': event, 'value': value, 'action': decision['action']})
        expected = 62
        regular = select_event(model, state, key, 'effective_previous', as_of=63)
        with patch('spora.internal_memory._decode_provenance', side_effect=experimental_decode):
            candidate = select_event(model, state, key, 'effective_previous', as_of=63)
        row = {'key': key, 'regular': regular, 'candidate': candidate,
               'all_writes_accepted': all(w['action'] in ('new', 'correct') for w in writes),
               'oracle_history_correct': historical(model, state, key, expected) == writes[expected]['value']}
        for label in ('regular', 'candidate'):
            output = row[label]
            row[label + '_correct'] = output['status'] == 'experimental_candidate' and output.get('event_time') == expected
            row[label + '_wrong_candidate'] = output['status'] == 'experimental_candidate' and output.get('event_time') != expected
            row[label + '_abstained'] = output['status'] == 'abstain'
        rows.append(row)
        print(json.dumps({'key': key, 'regular': regular['status'], 'candidate': candidate['status'],
                          'regular_correct': row['regular_correct'], 'candidate_correct': row['candidate_correct'],
                          'candidate_wrong': row['candidate_wrong_candidate']}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    summary = {'cases': len(rows), 'all_writes_accepted': sum(r['all_writes_accepted'] for r in rows),
               'oracle_history_correct': sum(r['oracle_history_correct'] for r in rows),
               **{label + '_' + field: sum(r[label + '_' + field] for r in rows)
                  for label in ('regular', 'candidate') for field in ('correct', 'wrong_candidate', 'abstained')}}
    report = {'status': 'experimental_confidence_band_selector_only', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Twelve synthetic dense same-key histories, 64 direct writes each.',
                              'New numeric keys, but patterns and confidence bands were chosen after earlier experiments.',
                              'Oracle assumes each direct write represents a true change and checks selected event index.',
                              'Only clean in-memory states; no compaction, corruption, delayed or uncertain sources.',
                              'Band change exists only during this local test; default decoder remains unchanged.',
                              'No natural-language input, answer generation or real user state evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
