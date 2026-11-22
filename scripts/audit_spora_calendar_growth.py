"""Exercise online calendar writes across SPORA history page boundaries."""

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import WORDS
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_calendar_history_chat import read_calendar
from spora.durable import TensorStore
from spora.integrated_durable import load_integrated_durable
from spora.language_training import frozen_digest


def page_tensors(state):
    return [tuple(tensor for site in page for tensor in site)
            + (state.internal.event_keys[i], state.internal.provenance[i], state.internal.provenance_precision[i])
            for i, page in enumerate(state.pages)]


def extend(runtime, key):
    first = runtime.state.cursor
    step_day = runtime.observation_calendar.days[-1]
    page_size = runtime.organism.config.time_count
    if (len(runtime.state.pages) != 1 or first != 6 or first >= page_size
            or runtime.state.internal.provenance_precision is None):
        raise ValueError('Expected the verified six-observation calendar fixture with independent provenance')
    other = [value for value in range(runtime.organism.config.key_count) if value != key]
    targets = {page_size: '豆浆', 2 * page_size: '茶'}
    boundaries = []
    for event in range(first, 2 * page_size + 3):
        day = step_day + timedelta(days=event - first + 1)
        value = targets.get(event)
        address = key if value is not None else other[event % len(other)]
        word = WORDS.index(value) if value is not None else event % len(WORDS)
        completed = len(runtime.state.pages)
        old = page_tensors(runtime.state) if event % page_size == 0 else ()
        decision, stats = runtime.observe_sourced(address, word, 'direct', 1., observed_on=day)
        if decision['history_time'] != event or runtime.observation_calendar.days[-1] != day:
            raise ValueError('Observation, date and history cursor diverged')
        if old:
            if (not decision['current_write'] or len(runtime.state.pages) != completed + 1
                    or any(not torch.equal(a, b) for before, after in zip(old, page_tensors(runtime.state)[:completed], strict=True)
                           for a, b in zip(before, after, strict=True))):
                raise ValueError('Page boundary changed old tensors or rejected the new value')
            boundaries.append({'event': event, 'day': day.isoformat(), 'value': value, 'write': stats})
            print(f'Event {event} opened page {completed + 1}', flush=True)
    return boundaries


def check(runtime, tokenizer, boundaries, *, state=None):
    state = runtime.state if state is None else state
    source_day = runtime.observation_calendar.days[5].isoformat()
    expected = [
        ('只按截至2026-01-04已收到的记录，2026-01-02对小林有效的饮品是什么？', 0, '咖啡'),
        ('帮我查小林在2026年1月2日当天登记的饮料是哪种？', 4, '果汁'),
        (f'小林在{source_day}的饮品是什么？', 5, '牛奶'),
    ]
    expected.extend((f'小林在{row["day"]}的饮品是什么？', row['event'], row['value']) for row in boundaries)
    rows = []
    for question, event, value in expected:
        result = read_calendar(runtime.decoder, tokenizer, runtime.organism, state,
                               runtime.observation_calendar, question)
        generated = result['generated'] or {}
        passed = (generated.get('event_time') == event and generated.get('matrix_value') == value
                  and (generated.get('answer') or '').rstrip('。！ ') == value)
        rows.append({'question': question, 'expected_event': event, 'expected_value': value,
                     'passed': passed, **result})
    return rows


def main(args):
    started = time.perf_counter()
    torch.set_num_threads(2)
    database = args.output / 'individual.sqlite'
    result_path = args.output / ('reopened.json' if args.reopen else 'result.json')
    if result_path.exists() or (not args.reopen and args.output.exists()):
        raise FileExistsError(result_path)
    if args.reopen:
        prior = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
    else:
        args.output.mkdir(parents=True)
        with TensorStore(args.source) as seed:
            seed.backup(database)
    source_sha = sha(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    with load_integrated_durable(database, args.checkpoint, load_full_streamed().eval(), MODEL) as runtime:
        key = runtime.organism.identity_fingerprints.lookup('小林')
        boundaries = prior['boundaries'] if args.reopen else extend(runtime, key)
        before = frozen_digest(runtime.organism, runtime.state)
        rows = check(runtime, tokenizer, boundaries)
        if before != frozen_digest(runtime.organism, runtime.state) or source_sha != sha(args.checkpoint):
            raise ValueError('Reading changed an individual or the decoder source')
        report = {'writes': runtime.state.cursor, 'pages': len(runtime.state.pages), 'revision': runtime.store.revision,
                  'source_sha256': source_sha, 'state_sha256': before, 'boundaries': boundaries, 'reads': rows}
        if args.reopen:
            if any(prior[key] != value for key, value in report.items()):
                raise ValueError('Independent reload changed the page growth result')
            report['exact_reopen'] = True
        report['seconds'] = time.perf_counter() - started
    write(result_path, report)
    print(f'Read {sum(row["passed"] for row in rows)}/{len(rows)} at {report["writes"]} observations', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reopen', action='store_true')
    args = parser.parse_args()
    if not args.reopen and args.source is None:
        parser.error('--source is required to create the audit copy')
    main(args)
