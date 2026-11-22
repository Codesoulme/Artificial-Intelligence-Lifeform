"""Compare effective-time and past-knowledge reads on the same saved individual."""

import argparse
import copy
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import KEYS, NAMES, WORDS, answer
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import load, save
from spora.historical_resolutions import HistoricalResolutions
from spora.integrated_checkpoint import load_integrated_runtime
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced
from spora.language_training import frozen_digest


EPOCH = date(2026, 1, 1)
CASES = (
    ('before_backfill', 1, 3, 0),
    ('after_backfill', 1, None, 4),
    ('before_correction_known', 2, 1, 0),
    ('after_correction_known', 2, 2, 2),
    ('visible_unresolved_backfill', 1, 4, None),
    ('outside_backfill_window', 0, 4, 0),
)


def stamp(day):
    return (EPOCH + timedelta(days=day)).isoformat()


def fixture(organism, target, distractor, values, output, tag):
    state = fresh_controlled_state(organism)
    state = enable_independent_provenance(organism, enable_event_key_index(organism, state))
    for event, (key, valid_at) in enumerate(((target, 0), (distractor, 1), (target, 2),
                                             (distractor, 3), (target, 1))):
        state, decision = observe_sourced(organism, state, key, values[event], 'direct', 1., valid_at=valid_at)
        if decision['action'] != ('new', 'new', 'correct', 'correct', 'defer')[event]:
            raise ValueError(f'Unexpected action at event {event}: {decision}')
        if event in (1, 2, 3, 4):
            save(output / f'{tag}-through-{event}.pt', organism, state)
    organism.historical_resolutions = HistoricalResolutions(organism.config.key_count, capacity=2)
    organism.historical_resolutions.record(organism, state, target, 4, value=values[4], valid_at=1, decision='confirm')
    save(output / f'{tag}-present.pt', organism, state)


def audit(runtime, tokenizer, output, *, reopen):
    rows, digests = [], {}
    for index in (0, 3):
        key, name = KEYS[index], NAMES[index]
        for shift in (0, 2):
            tag = f'person-{index}-world-{shift}'
            values = tuple((number + shift) % len(WORDS) for number in (1, 2, 6, 5, 3))
            if not reopen:
                fixture(copy.deepcopy(runtime.organism), key, KEYS[1], values, output, tag)
            organism, state, _ = load(output / f'{tag}-present.pt', 'cpu')
            digests[tag] = frozen_digest(organism, state)
            for label, effective, known_by, expected in CASES:
                known = f'只按截至{stamp(known_by)}已收到的记录，' if known_by is not None else '按现在掌握的完整记录，'
                question = f'{known}{stamp(effective)}对{name}有效的饮品是什么？只回答饮品名称。'
                result = select_event(organism, state, key, 'effective_latest', as_of=effective, known_by=known_by)
                reference = None
                if known_by is not None:
                    earlier, snapshot, _ = load(output / f'{tag}-through-{known_by}.pt', 'cpu')
                    reference = select_event(earlier, snapshot, key, 'effective_latest', as_of=effective)
                generated = None
                if result['status'] == 'experimental_candidate':
                    generated = answer(runtime.decoder, tokenizer, organism, question, state, key, 2,
                                       event=result['event_time'], use_cache=True, output_gain=1.5)
                expected_word = None if expected is None else WORDS[values[expected]]
                correct_route = ((result.get('reason') == 'unresolved_factual_observation') if expected is None else
                                 (result.get('event_time') == expected and result['status'] == 'experimental_candidate'))
                correct_answer = (generated is None if expected is None else generated is not None and
                                  generated.get('matrix_value') == expected_word and
                                  (generated.get('answer') or '').rstrip('。！ ') == expected_word)
                rows.append({'case': label, 'person': name, 'world': shift, 'question': question,
                             'expected_event': expected, 'expected_value': expected_word, 'selection': result,
                             'earlier_snapshot': reference, 'generated': generated,
                             'correct_route': correct_route, 'correct_answer': correct_route and correct_answer})
            if frozen_digest(organism, state) != digests[tag]:
                raise ValueError('Historical read changed a saved individual')
    return rows, digests


def main(checkpoint, output, *, reopen=False):
    destination = output / ('reopened.json' if reopen else 'result.json')
    if destination.exists() or (not reopen and output.exists()):
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_sha = sha(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if not reopen:
        output.mkdir(parents=True, exist_ok=False)
    rows, digests = audit(runtime, tokenizer, output, reopen=reopen)
    snapshots = [row for row in rows if row['earlier_snapshot'] is not None]
    summary = {'queries': len(rows), 'right_event_or_abstention': sum(row['correct_route'] for row in rows),
               'right_value_or_abstention': sum(row['correct_answer'] for row in rows),
               'snapshot_comparisons': len(snapshots),
               'snapshot_agreement': sum(row['selection'] == row['earlier_snapshot'] for row in snapshots)}
    report = {'summary': summary, 'rows': rows, 'state_sha256': digests, 'source_sha256': source_sha}
    if reopen:
        first = json.loads((output / 'result.json').read_text(encoding='utf-8'))
        if any(first[key] != value for key, value in report.items()):
            raise ValueError('Independent process produced different historical reads')
        report['exact_reopen'] = True
    if sha(checkpoint) != source_sha or frozen_digest(runtime.organism, runtime.state) != individual:
        raise ValueError('The audit modified a source')
    report.update(seconds=time.perf_counter() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    write(destination, report)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reopen', action='store_true')
    args = parser.parse_args()
    main(args.checkpoint, args.output, reopen=args.reopen)
