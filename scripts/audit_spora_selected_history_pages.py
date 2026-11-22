"""Measure explicit history reads with eager and selected-bank snapshots."""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from statistics import median
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_spora_calendar_growth import check
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_durable import load_integrated_durable
from spora.language_training import frozen_digest
from spora.sleep_memory import SealedPage, history_site


def mode(decoder, eager):
    if not eager:
        return nullcontext()
    return patch.object(decoder, '_selected_memory_banks',
                        new=lambda slots, _mask, count: (range(count), slots.detach().clone()))


@torch.no_grad()
def snapshot_benchmark(runtime):
    state = runtime.state
    sealed = [page for page in state.pages if isinstance(page, SealedPage)]
    device = state.current[0].device
    mask = torch.ones(1, dtype=torch.bool, device=device)
    results = []
    for page_count in (1, 3, 8, 16):
        sample = replace(state, pages=[sealed[index % len(sealed)] for index in range(page_count)])
        for bank in (0, page_count):
            slots = torch.tensor([bank], dtype=torch.long, device=device)
            for eager in (True, False):
                with mode(runtime.decoder, eager):
                    with patch('spora.integrated_open_base.history_site', wraps=history_site) as materialize:
                        with runtime.decoder._snapshot(sample, None, False, mask, slots):
                            pass
                        calls = materialize.call_count
                    durations = []
                    for index in range(60):
                        start = time.perf_counter_ns()
                        with runtime.decoder._snapshot(sample, None, False, mask, slots):
                            pass
                        elapsed = (time.perf_counter_ns() - start) / 1000
                        if index >= 10:
                            durations.append(elapsed)
                results.append({'sealed_pages': page_count, 'bank': 'history' if bank else 'current',
                                'method': 'eager' if eager else 'selected', 'history_site_calls': calls,
                                'snapshot_median_us': median(durations), 'snapshots': len(durations)})
    return results


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    baseline = json.loads((args.sleep / 'reopened.json').read_text(encoding='utf-8'))
    growth = json.loads((args.growth / 'result.json').read_text(encoding='utf-8'))
    source = args.sleep / 'individual.sqlite'
    protected = {str(path): sha(path) for path in (source, args.checkpoint)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    with load_integrated_durable(source, args.checkpoint, load_full_streamed().eval(), MODEL) as runtime:
        before = frozen_digest(runtime.organism, runtime.state)
        if not baseline.get('exact_reopen') or before != baseline['state_sha256']:
            raise ValueError('The benchmark needs the verified, reopened sleep individual')
        boundaries = [*growth['boundaries'], baseline['wake']]
        for index, eager in enumerate((True, False, True, False, False, True, True, False)):
            with mode(runtime.decoder, eager):
                torch.cuda.synchronize()
                start = time.perf_counter()
                rows = check(runtime, tokenizer, boundaries)
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
            if rows != baseline['reads']:
                raise ValueError('The calendar answers or selected events changed')
            if index >= 2:
                method = 'eager' if eager else 'selected'
                result = results.setdefault(method, {'seconds_repeats': [], 'rows': rows})
                result['seconds_repeats'].append(duration)
        for result in results.values():
            result['seconds'] = median(result['seconds_repeats'])
        samples = snapshot_benchmark(runtime)
        if before != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('The benchmark changed the individual')
    if any(sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('The benchmark changed a saved source')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, {'real': results, 'snapshot_cpu': samples, 'sources': protected})
    for row in samples:
        print(row, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('checkpoint', 'growth', 'sleep', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    main(parser.parse_args())
