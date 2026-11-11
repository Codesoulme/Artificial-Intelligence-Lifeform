"""Trace uncertain provenance in a controlled, interleaved multi-page history."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_delayed_fact_audit import SCENARIOS, load_individual_from_verified_checkpoint
from scripts.deepseek_spora_history_scan_scaling import OUTPUT as PRIOR
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, query
from spora.historical_resolutions import HistoricalResolutions
from spora.internal_memory import observe_sourced, read_provenance_batch
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-cross-key-history-probe-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if (prior['source_checkpoint_sha256'] != source_sha or prior['rows'][2]['noise_writes'] != 128
            or prior['rows'][2]['selected'].get('reason') != 'uncertain_provenance'):
        raise ValueError('Prior uncertainty evidence mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    model.historical_resolutions = HistoricalResolutions(model.config.key_count, 4)
    state = fresh_controlled_state(model)
    target, distractor = ENROLLMENTS[0][0], ENROLLMENTS[3][0]
    observed = []
    for subject, value, valid_at in SCENARIOS[1]:
        key = target if subject == 'T' else distractor
        state, decision = observe_sourced(model, state, key, value, 'direct', 1.0, valid_at=valid_at)
        observed.append({'key': key, 'value': value, 'action': decision['action']})
    model.historical_resolutions.record(model, state, target, 4, value=3, valid_at=1, decision='confirm')
    noise_keys = [key for key in range(10, 51) if key not in (target, distractor)]
    for i in range(128):
        if state.cursor == capacity(model, state):
            state = grow(model, state)
        key = noise_keys[i % len(noise_keys)]
        value = (i * 7 + 2) % model.config.unknown
        state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
        observed.append({'key': key, 'value': value, 'action': decision['action']})
    events = list(range(state.cursor))
    provenance = read_provenance_batch(model, state, [(target, event) for event in events], batch_size=16)
    rows = []
    for start in range(0, len(events), 16):
        batch = events[start:start + 16]
        keys = torch.full((len(batch),), target, dtype=torch.long)
        times = torch.tensor(batch, dtype=torch.long)
        with torch.no_grad():
            logits = query(model, state, keys, torch.ones_like(keys), times)
            probs = logits.float().softmax(-1)
            values, indices = probs.topk(2, dim=-1)
        for j, event in enumerate(batch):
            row = {'event': event, 'is_target': observed[event]['key'] == target,
                   'observed_key': observed[event]['key'], 'observed_value': observed[event]['value'],
                   'target_top': int(indices[j, 0]), 'target_confidence': float(values[j, 0]),
                   'target_margin': float(values[j, 0] - values[j, 1]), 'provenance': provenance[event]}
            row['target_known'] = row['target_top'] != model.config.unknown
            rows.append(row)
    selected = select_event(model, state, target, 'effective_previous', as_of=state.cursor - 1)
    first_problem = selected.get('event_time')
    if selected.get('reason') != 'uncertain_provenance' or type(first_problem) is not int:
        raise ValueError('Original uncertain-provenance behavior did not reproduce')
    true_events = [r for r in rows if r['is_target']]
    other_events = [r for r in rows if not r['is_target']]
    summary = {'events': len(rows), 'true_target_events': len(true_events),
               'true_target_predicted_known': sum(r['target_known'] for r in true_events),
               'other_events': len(other_events), 'other_predicted_known': sum(r['target_known'] for r in other_events),
               'other_uncertain_provenance': sum('reason' in r['provenance']
                                                 and r['provenance']['reason'] != 'unknown_event' for r in other_events),
               'first_problem_event': first_problem, 'first_problem_is_target': rows[first_problem]['is_target'],
               'first_problem': rows[first_problem]}
    del model.historical_resolutions
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    result = {'status': 'controlled_cross_key_history_probe', 'summary': summary, 'rows': rows,
              'selection': selected, 'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['One handcrafted trajectory, one target key, and deterministic unrelated writes.',
                              'Per-event target-known uses argmax, not a calibrated probability.',
                              'Ground truth of written key/value comes from the controlled experiment.',
                              'Provenance uncertainty can have several causes; this does not identify all of them.',
                              'No real user state, natural language, or generation evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
