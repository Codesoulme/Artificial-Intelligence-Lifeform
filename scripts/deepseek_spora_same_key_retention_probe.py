"""Separate value and provenance recall in a dense same-key trajectory."""
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
from scripts.deepseek_spora_delayed_fact_audit import load_individual_from_verified_checkpoint
from scripts.deepseek_spora_same_key_density_audit import OUTPUT as PRIOR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, query
from spora.internal_memory import enable_event_key_index, observe_sourced, read_provenance_batch
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-same-key-retention-probe-20261112.json'
LEVELS = (8, 32, 64, 128, 256)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    previous = json.loads(PRIOR.read_text(encoding='utf-8'))
    if previous['source_checkpoint_sha256'] != source_sha:
        raise ValueError('Prior checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    state = enable_event_key_index(model, fresh_controlled_state(model))
    key = ENROLLMENTS[0][0]
    count, writes, summaries = 0, [], []
    for level in LEVELS:
        while count < level:
            if state.cursor == capacity(model, state):
                state = grow(model, state)
            value = 1 if count % 2 == 0 else 6
            state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
            writes.append({'value': value, 'action': decision['action'], 'event': count})
            count += 1
        items = read_provenance_batch(model, state, [(key, i) for i in range(count)], batch_size=16)
        event_rows = []
        for start in range(0, count, 16):
            events = list(range(start, min(count, start + 16)))
            keys = torch.full((len(events),), key, dtype=torch.long)
            times = torch.tensor(events, dtype=torch.long)
            with torch.no_grad():
                tops = query(model, state, keys, torch.ones_like(keys), times).argmax(-1).tolist()
            for event, top in zip(events, tops):
                observed = items[event]
                expected = writes[event]
                event_rows.append({'event': event, 'expected_value': expected['value'], 'predicted_value': top,
                                   'value_correct': top == expected['value'], 'expected_action': expected['action'],
                                   'provenance': observed,
                                   'provenance_correct': (observed.get('source') == 'direct'
                                                          and observed.get('action') == expected['action']
                                                          and observed.get('valid_at') == event and 'reason' not in observed)})
        summary = {'writes': level, 'pages': len(state.pages),
                   'value_correct': sum(r['value_correct'] for r in event_rows),
                   'value_unknown': sum(r['predicted_value'] == model.config.unknown for r in event_rows),
                   'provenance_correct': sum(r['provenance_correct'] for r in event_rows),
                   'uncertain_provenance': sum(r['provenance'].get('reason') not in (None, 'unknown_event') for r in event_rows),
                   'unknown_provenance': sum(r['provenance'].get('reason') == 'unknown_event' for r in event_rows),
                   'rows': event_rows}
        summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    report = {'status': 'same_key_neural_retention_probe', 'levels': summaries,
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Single known key, deterministic alternating values and direct numerical writes.',
                              'Oracle uses actual write decisions for provenance action and event index for effective time.',
                              'Measures native value argmax and decoded provenance, not language output.',
                              'No independent random trajectories, power failures or real user state.',
                              'Exact key index identifies events; values and provenance remain neural.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'levels': len(summaries)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
