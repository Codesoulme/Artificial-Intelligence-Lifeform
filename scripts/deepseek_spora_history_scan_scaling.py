"""Measure a confirmed event scan as a controlled history grows."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_delayed_fact_audit import SCENARIOS, current, load_individual_from_verified_checkpoint
from scripts.deepseek_spora_resolution_interference_audit import OUTPUT as PRIOR
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow
from spora.historical_resolutions import HistoricalResolutions
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.state_codec import flatten

OUTPUT = ROOT / 'runs/deepseek-spora-history-scan-scaling-20261112.json'
LEVELS = (0, 64, 128, 256, 512)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != source_sha or prior['summary']['64']['selection_correct'] != 6:
        raise ValueError('Prior interference evidence mismatch')
    torch.set_num_threads(2)
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    model.historical_resolutions = HistoricalResolutions(model.config.key_count, capacity=4)
    state = fresh_controlled_state(model)
    key, distractor = ENROLLMENTS[0][0], ENROLLMENTS[3][0]
    for subject, value, valid_at in SCENARIOS[1]:
        state, _ = observe_sourced(model, state, key if subject == 'T' else distractor,
                                   value, 'direct', 1.0, valid_at=valid_at)
    model.historical_resolutions.record(model, state, key, 4, value=3, valid_at=1, decision='confirm')
    noise_keys = [k for k in range(10, 51) if k not in (key, distractor)]
    count, rows = 0, []
    for level in LEVELS:
        while count < level:
            if state.cursor == capacity(model, state):
                state = grow(model, state)
            state, _ = observe_sourced(model, state, noise_keys[count % len(noise_keys)],
                                        (count * 7 + 2) % model.config.unknown, 'direct', 1.0)
            count += 1
        times, outputs = [], []
        for _ in range(3):
            start = time.perf_counter()
            result = select_event(model, state, key, 'effective_previous', as_of=state.cursor - 1)
            times.append(time.perf_counter() - start)
            outputs.append(result)
        if not all(result == outputs[0] for result in outputs):
            raise ValueError('Repeated selection output changed')
        _, tensors = flatten(model, state)
        tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
        selected = outputs[0]
        row = {'noise_writes': level, 'cursor': state.cursor, 'pages': len(state.pages),
               'tensor_bytes': tensor_bytes, 'scan_seconds': times, 'scan_median_seconds': statistics.median(times),
               'current_correct': current(model, state, key) == 6,
               'selected': selected,
               'correct': selected['status'] == 'experimental_candidate' and selected.get('event_time') == 4}
        rows.append(row)
        print(json.dumps({'noise': level, 'cursor': state.cursor, 'pages': len(state.pages),
                          'tensor_bytes': tensor_bytes, 'median_seconds': row['scan_median_seconds'],
                          'correct': row['correct'], 'current_correct': row['current_correct']}), flush=True)
    del model.historical_resolutions
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    report = {'status': 'controlled_cpu_history_scan_scaling', 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'torch_threads': 2,
              'limitations': ['One synthetic history; not a latency distribution across real workloads.',
                              'CPU selector timing excludes language-model generation, I/O and checkpoint save.',
                              'Tensor bytes sum serialized tensors without file container overhead or compression.',
                              'Fixed handcrafted noise sequence and one known, manually confirmed event.',
                              'Three repeated measurements per level; no warmup calibration or system isolation.',
                              'No natural-language learning or actual user individual.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'levels': len(rows), 'all_correct': all(r['correct'] for r in rows)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
