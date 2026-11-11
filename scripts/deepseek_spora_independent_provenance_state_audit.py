"""Exercise the actual opt-in provenance writer on dense two-page states."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_delayed_fact_audit import historical, load_individual_from_verified_checkpoint
from scripts.deepseek_spora_provenance_band_selector_pilot import KEYS, OUTPUT as PRIOR
from scripts.deepseek_spora_provenance_ridge_holdout import OUTPUT as RIDGE_EVIDENCE
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, save
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-independent-provenance-state-audit-20261112.json'
LEVELS = (64, 128)


def equal_value_state(a, b):
    if a.cursor != b.cursor or len(a.pages) != len(b.pages):
        return False
    pairs = list(zip(a.current, b.current)) + [(a.internal.evidence, b.internal.evidence)]
    for pa, pb in zip(a.pages, b.pages):
        for sa, sb in zip(pa, pb):
            pairs.extend(zip(sa, sb))
    pairs.extend(zip(a.internal.event_keys, b.internal.event_keys))
    return all(torch.equal(x, y) for x, y in pairs)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    for path in (PRIOR, RIDGE_EVIDENCE):
        if json.loads(path.read_text(encoding='utf-8'))['source_checkpoint_sha256'] != source_sha:
            raise ValueError('Prior checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    rows = []
    for key in KEYS:
        regular = enable_event_key_index(model, fresh_controlled_state(model))
        candidate = enable_independent_provenance(model, enable_event_key_index(model, fresh_controlled_state(model)))
        accepted = 0
        for event in range(max(LEVELS)):
            if regular.cursor == capacity(model, regular):
                regular, candidate = grow(model, regular), grow(model, candidate)
            value = (1, 6, 3, 4)[(event + key) % 4]
            regular, a = observe_sourced(model, regular, key, value, 'direct', 1.0)
            candidate, b = observe_sourced(model, candidate, key, value, 'direct', 1.0)
            if a != b:
                raise ValueError('Write decisions diverged')
            accepted += int(a['current_write'])
            if event + 1 not in LEVELS:
                continue
            if not equal_value_state(regular, candidate):
                raise ValueError('Value state changed under alternate provenance')
            expected_event = event - 1
            expected_value = (1, 6, 3, 4)[(expected_event + key) % 4]
            old = select_event(model, regular, key, 'effective_previous', as_of=event)
            new = select_event(model, candidate, key, 'effective_previous', as_of=event)
            row = {'key': key, 'writes': event + 1, 'accepted_writes': accepted,
                   'expected_previous_event': expected_event, 'oracle_value_correct':
                   historical(model, candidate, key, expected_event) == expected_value,
                   'regular': old, 'candidate': new,
                   'regular_correct': old['status'] == 'experimental_candidate' and old.get('event_time') == expected_event,
                   'candidate_correct': new['status'] == 'experimental_candidate' and new.get('event_time') == expected_event,
                   'candidate_wrong': new['status'] == 'experimental_candidate' and new.get('event_time') != expected_event,
                   'candidate_abstained': new['status'] == 'abstain',
                   'precision_bytes': sum(p.numel() * p.element_size() for p in candidate.internal.provenance_precision)}
            rows.append(row)
            print(json.dumps({'key': key, 'writes': event + 1, 'regular_correct': row['regular_correct'],
                              'candidate_correct': row['candidate_correct'], 'candidate_wrong': row['candidate_wrong'],
                              'candidate_reason': new.get('reason')}), flush=True)
        if key == KEYS[0]:
            digest = frozen_digest(model, candidate)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'individual.pt'
                save(path, model, candidate, {'purpose': 'independent-provenance-pilot'})
                restored_model, restored, _ = load(path, 'cpu')
                reload_result = select_event(restored_model, restored, key, 'effective_previous', as_of=127)
                reload_same = (frozen_digest(restored_model, restored) == digest and reload_result == rows[-1]['candidate'])
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    summary = {}
    for level in LEVELS:
        subset = [r for r in rows if r['writes'] == level]
        summary[str(level)] = {'cases': len(subset), 'regular_correct': sum(r['regular_correct'] for r in subset),
                               'candidate_correct': sum(r['candidate_correct'] for r in subset),
                               'candidate_wrong': sum(r['candidate_wrong'] for r in subset),
                               'candidate_abstained': sum(r['candidate_abstained'] for r in subset),
                               'oracle_value_correct': sum(r['oracle_value_correct'] for r in subset),
                               'all_writes_accepted': sum(r['accepted_writes'] == level for r in subset)}
    report = {'status': 'opt_in_independent_provenance_state_audit', 'summary': summary, 'rows': rows,
              'same_process_reload_same': reload_same, 'source_checkpoint_sha256': source_sha,
              'seconds': time.perf_counter() - started,
              'limitations': ['Twelve previously examined numeric keys with deterministic four-value changes.',
                              'The original selector questions and most keys were used in earlier development.',
                              'Explicit direct numeric observations, not real language, and no decoder answers.',
                              'Independent precision is opt-in and only for fresh controlled states.',
                              'Same-process temporary reload is not crash or multi-process recovery.',
                              'Original and candidate retain the same conservative provenance thresholds.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'summary': summary, 'reload_same': reload_same}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
